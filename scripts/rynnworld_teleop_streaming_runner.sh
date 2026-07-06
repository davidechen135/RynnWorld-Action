#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# rynnworld_teleop_streaming_runner.sh — Core launcher for both training phases.
#
# Auto-detects single-node vs multi-node from environment variables, renders
# the accelerate config, and launches the training script via accelerate.
#
# This script is invoked by the phase-specific wrappers (which set phase-
# specific env defaults) and is not usually called directly.
#
# Multi-node usage (set by your cluster scheduler):
#   WORLD_SIZE         total number of nodes
#   RANK               this node's rank in [0, WORLD_SIZE)
#   MASTER_ADDR        rank-0 hostname
#   MASTER_PORT        rank-0 port
#   NPROC_PER_NODE     GPUs per node
#
# Single-node usage (no setup needed): the script auto-detects local GPUs.
#
# Local quick-test (short run on one machine):
#   WORLD_SIZE=1 RANK=0 NPROC_PER_NODE=8 \
#     MASTER_ADDR=localhost MASTER_PORT=29721 \
#     NUM_TRAIN_STEPS=20 DECODE_EVERY=10 SAVE_STEPS=20 \
#     bash scripts/train_runner.sh
# ─────────────────────────────────────────────────────────────────────────────
{
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

# ── Ensure a few small deps are installed (big packages like torch/diffusers assumed present) ──
if [ "${SKIP_PIP_INSTALL:-0}" != "1" ]; then
    pip install --quiet pyyaml safetensors termcolor imageio imageio-ffmpeg 2>/dev/null
fi

# ── Auto-load config file ──
# Precedence: command-line env > config file > script defaults.
# Override path via CONFIG env var; otherwise looks for configs/config.yaml.
CONFIG_PATH="${CONFIG:-${REPO_DIR}/configs/config.yaml}"
if [ -f "$CONFIG_PATH" ]; then
    eval "$(python3 "${SCRIPT_DIR}/streaming_load_config.py" "$CONFIG_PATH" 2>&1)"
fi

# ── Auto-detect single-node vs multi-node ──
# If platform set the multi-node env vars, use them. Otherwise default to
# single-node (1 machine, all local GPUs). This way the same script works
# both on a single workstation (no setup) and on a multi-node cluster.
if [ -z "${WORLD_SIZE:-}" ] || [ -z "${RANK:-}" ] || [ -z "${MASTER_ADDR:-}" ] \
    || [ -z "${MASTER_PORT:-}" ] || [ -z "${NPROC_PER_NODE:-}" ]; then
    # Auto-detect GPU count for single-node mode
    if command -v nvidia-smi >/dev/null 2>&1; then
        _AUTO_NPROC=$(nvidia-smi --query-gpu=count --format=csv,noheader 2>/dev/null | head -1)
        [ -z "$_AUTO_NPROC" ] && _AUTO_NPROC=$(nvidia-smi -L 2>/dev/null | wc -l)
    fi
    [ -z "${_AUTO_NPROC:-}" ] || [ "${_AUTO_NPROC}" -eq 0 ] && _AUTO_NPROC=8

    export WORLD_SIZE="${WORLD_SIZE:-1}"
    export RANK="${RANK:-0}"
    export MASTER_ADDR="${MASTER_ADDR:-localhost}"
    export MASTER_PORT="${MASTER_PORT:-29500}"
    export NPROC_PER_NODE="${NPROC_PER_NODE:-${_AUTO_NPROC}}"

    echo "═══════════════════════════════════════════════════════════════════════════════"
    echo "  Single-node mode (auto-detected)"
    echo "    WORLD_SIZE=${WORLD_SIZE}  RANK=${RANK}  NPROC_PER_NODE=${NPROC_PER_NODE}"
    echo "    MASTER_ADDR=${MASTER_ADDR}  MASTER_PORT=${MASTER_PORT}"
    echo "  For multi-node, set WORLD_SIZE/RANK/MASTER_ADDR/MASTER_PORT/NPROC_PER_NODE explicitly."
    echo "═══════════════════════════════════════════════════════════════════════════════"
fi

TOTAL_GPUS=$((WORLD_SIZE * NPROC_PER_NODE))
# Export so sed (for configs_acc/streaming_multinode.yaml) can see ${TOTAL_GPUS}.
# accelerate interprets num_processes as the TOTAL rank count (not per-node),
# so the rendered yaml must use ${TOTAL_GPUS}.
export TOTAL_GPUS

# ── Save cluster's notion of "WORLD_SIZE / RANK / NPROC_PER_NODE" before
#    accelerate launch overwrites them ──
# accelerate launch (via torchrun-style) RESETS these env vars in the child
# Python process to:
#   WORLD_SIZE      = TOTAL_GPUS (global rank count)
#   RANK            = global rank ∈ [0, TOTAL_GPUS)
#   LOCAL_RANK      = local rank within node ∈ [0, NPROC_PER_NODE)
#   LOCAL_WORLD_SIZE= NPROC_PER_NODE
# This SHADOWS the cluster's original semantics (WORLD_SIZE=#machines,
# RANK=node_rank). Train script's [MULTINODE-CHECK] needs to know the
# CLUSTER-level values to verify multi-node init.
# Fix: save them to CLUSTER_* prefixed var names that torchrun won't touch.
export CLUSTER_WORLD_SIZE="$WORLD_SIZE"        # number of MACHINES
export CLUSTER_NPROC_PER_NODE="$NPROC_PER_NODE" # GPUs per machine
export CLUSTER_RANK="$RANK"                     # this machine's rank ∈ [0, WS)
export CLUSTER_TOTAL_GPUS="$TOTAL_GPUS"         # total expected ranks

# ── Cross-cluster .pyc safety ──
# Same as single-node: drop stale .pyc that may have been written by an older
# Python image from a different host's filesystem.
for _pycache in "${REPO_DIR}/__pycache__" "${REPO_DIR}/scripts/__pycache__"; do
    [ -d "$_pycache" ] && rm -rf "$_pycache"
done
export PYTHONDONTWRITEBYTECODE=1

# ════════════════════════════════════════════════════════════════════════════
# Early environment self-check
# Running the NCCL config + IB self-check up front means that if InfiniBand is
# unavailable we know immediately, instead of spending 5-10 minutes copying
# tens of GB of weights only to discover training will be 10x slower.
# ════════════════════════════════════════════════════════════════════════════

# ── NCCL config (required for multi-node: needs an inter-node interface) ──
# Default tries common cluster interface names. Override via NCCL_SOCKET_IFNAME
# env var to match your network setup.
# Fallback order: storage_bond → bond0 → eth0
if [ -z "${NCCL_SOCKET_IFNAME:-}" ]; then
    for _candidate in storage_bond bond0 eth0; do
        if [ -d "/sys/class/net/${_candidate}" ]; then
            NCCL_SOCKET_IFNAME="${_candidate}"
            break
        fi
    done
fi
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_CROSS_NIC="${NCCL_CROSS_NIC:-1}"
export NCCL_IB_TIMEOUT="${NCCL_IB_TIMEOUT:-22}"
[ -n "${NCCL_SOCKET_IFNAME:-}" ] && export NCCL_SOCKET_IFNAME

# ── IB / RDMA self-check ──
# Goals:
# 1. Search common locations for libibverbs and prepend its dir to
#    LD_LIBRARY_PATH so NCCL's dlopen can find it.
# 2. Print self-check results (rank 0 only, to avoid log spam), including
#    whether libibverbs loads + /sys/class/infiniband port status.
# 3. Non-destructive: if libibverbs is missing we just emit a warning and let
#    NCCL fall back to TCP — same behavior as before, just clearer logs.
if [ "${RANK}" = "0" ]; then
    echo "[ibcheck] Searching for libibverbs.so on this node..."
fi
_libibverbs_dirs=(
    "/usr/lib/x86_64-linux-gnu"
    "/usr/lib64"
    "/usr/lib"
    "/opt/conda/lib"
    "/usr/local/conda/lib"
    "${CONDA_PREFIX:-}/lib"
)
_libibverbs_found=""
for _d in "${_libibverbs_dirs[@]}"; do
    [ -z "$_d" ] && continue
    if [ -e "$_d/libibverbs.so.1" ] || [ -e "$_d/libibverbs.so" ]; then
        _libibverbs_found="$_d"
        # Prepend to LD_LIBRARY_PATH so NCCL's dlopen can find it
        export LD_LIBRARY_PATH="${_d}:${LD_LIBRARY_PATH:-}"
        break
    fi
done

# Also search via ldconfig cache (catches /etc/ld.so.conf.d/* paths)
if [ -z "$_libibverbs_found" ] && command -v ldconfig >/dev/null 2>&1; then
    _ld_path=$(ldconfig -p 2>/dev/null | awk '/libibverbs\.so/ {print $NF; exit}')
    if [ -n "$_ld_path" ] && [ -e "$_ld_path" ]; then
        _libibverbs_found="$(dirname "$_ld_path")"
        export LD_LIBRARY_PATH="${_libibverbs_found}:${LD_LIBRARY_PATH:-}"
    fi
fi

# Verify dlopen actually works (file exists vs ABI compatible are different).
# Use Python ctypes — same dlopen NCCL uses.
_libibverbs_loadable=0
if command -v python3 >/dev/null 2>&1; then
    if python3 -c "import ctypes; ctypes.CDLL('libibverbs.so.1')" 2>/dev/null; then
        _libibverbs_loadable=1
    fi
fi

# IB device port status check
_ib_active_count=0
_ib_total_count=0
if [ -d /sys/class/infiniband ]; then
    for _ib in /sys/class/infiniband/*/ports/1/state; do
        [ -e "$_ib" ] || continue
        _ib_total_count=$((_ib_total_count + 1))
        if grep -q "ACTIVE" "$_ib" 2>/dev/null; then
            _ib_active_count=$((_ib_active_count + 1))
        fi
    done
fi

# rank 0 only: print diagnostic banner
if [ "${RANK}" = "0" ]; then
    echo "═══════════════════════════════════════════════════════════════════════════════"
    if [ -n "$_libibverbs_found" ]; then
        echo "[ibcheck] ✅ libibverbs found at: $_libibverbs_found"
        echo "[ibcheck]    LD_LIBRARY_PATH prepended: $_libibverbs_found"
    else
        echo "[ibcheck] ❌ libibverbs.so NOT found in any known path!"
        echo "[ibcheck]    NCCL will fall back to TCP socket (storage_bond)"
        echo "[ibcheck]    Expected slowdown: ~5-10x for ZeRO-3 multi-node training"
        echo "[ibcheck]    Searched: ${_libibverbs_dirs[*]}"
        echo "[ibcheck]    Try: conda install -c conda-forge libibverbs"
        echo "[ibcheck]    Or:  apt install -y libibverbs1 (needs root)"
    fi
    if [ "$_libibverbs_loadable" = "1" ]; then
        echo "[ibcheck] ✅ python3 ctypes.CDLL(libibverbs.so.1) succeeded — NCCL should use IB"
    else
        echo "[ibcheck] ❌ python3 ctypes.CDLL(libibverbs.so.1) FAILED — NCCL will fall back to TCP"
        echo "[ibcheck]    Even if file was found, ABI / version may be incompatible."
    fi
    echo "[ibcheck] IB ports: $_ib_active_count ACTIVE / $_ib_total_count total"
    if [ "$_ib_total_count" -gt 0 ] && [ "$_ib_active_count" -eq 0 ]; then
        echo "[ibcheck] ⚠️  IB devices exist but all DOWN — check fabric/cable status"
    fi
    echo "[ibcheck] After launch, search 'libibverbs' in train.log:"
    echo "[ibcheck]   ✅ if NO 'Failed to open libibverbs.so' lines"
    echo "[ibcheck]   ✅ if 'NET/IB' (not 'NET/Socket') in 'Using network' log lines"
    echo "[ibcheck] ── PROCEEDING TO LAUNCH ──"
    echo "═══════════════════════════════════════════════════════════════════════════════"
fi
unset _libibverbs_dirs _libibverbs_found _libibverbs_loadable _ib_active_count _ib_total_count _d _ib _ld_path

# ── Base model path ──
# Users set MODEL_PATH directly (via env var or config.yaml). No prefetch —
# the training script reads weights straight from the given path.
if [ -z "${MODEL_PATH:-}" ]; then
    echo "ERROR: MODEL_PATH is not set. Export MODEL_PATH=<path-to-Wan2.2-TI2V-5B-Diffusers> or set it in config.yaml." >&2
    exit 1
fi
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: MODEL_PATH=$MODEL_PATH is not a directory." >&2
    exit 1
fi
echo "[runner] [rank ${RANK:-0}] base model: $MODEL_PATH"

# ── Teacher checkpoint ──
# Users set TEACHER_CKPT directly (via env var or config.yaml). No prefetch —
# the training script reads weights straight from the given path.
# Set TEACHER_CKPT=none (or leave unset) to skip teacher loading entirely.
USE_EMA_TEACHER="${USE_EMA_TEACHER:-0}"
TEACHER_ARG=""

if [ -z "${TEACHER_CKPT:-}" ] || [ "${TEACHER_CKPT}" = "none" ]; then
    TEACHER_CKPT="none"
    echo "[runner] [rank ${RANK:-0}] TEACHER_CKPT=none → student starts from base model only"
else
    if [ ! -d "${TEACHER_CKPT}" ]; then
        echo "ERROR: TEACHER_CKPT=${TEACHER_CKPT} is not a directory." >&2
        exit 1
    fi
    echo "[runner] [rank ${RANK:-0}] teacher ckpt: $TEACHER_CKPT"
    TEACHER_ARG="--teacher_ckpt ${TEACHER_CKPT}"
fi

if [ -n "${TEACHER_ARG}" ] && [ "${USE_EMA_TEACHER}" = "1" ]; then
    TEACHER_ARG="${TEACHER_ARG} --use_ema_teacher"
    if [ ! -f "${TEACHER_CKPT}/ema_weights.bin" ] && [ ! -f "${TEACHER_CKPT}/ema_weights.pt" ]; then
        echo "WARNING: ${TEACHER_CKPT}/ema_weights.{bin,pt} missing — fallback per load_teacher_into_pipe logic" >&2
    fi
fi

# ── Optional --teacher_init_from_checkpoint and --teacher_control_type args ──
# When the teacher checkpoint is an adapter (e.g. LoRA) trained on top of a
# separate base SFT, --teacher_init_from_checkpoint points at that base SFT so
# the teacher's frozen weights can be loaded correctly.
if [ -n "${TEACHER_INIT_FROM_CHECKPOINT:-}" ]; then
    if [ ! -d "${TEACHER_INIT_FROM_CHECKPOINT}" ]; then
        echo "ERROR: TEACHER_INIT_FROM_CHECKPOINT=${TEACHER_INIT_FROM_CHECKPOINT} is not a directory." >&2
        exit 1
    fi
    TEACHER_ARG="${TEACHER_ARG} --teacher_init_from_checkpoint ${TEACHER_INIT_FROM_CHECKPOINT}"
fi

# Default control_type matches the standard teacher recipe; override per-phase
# from the wrapper if the teacher uses a different control formulation.
TEACHER_CONTROL_TYPE="${TEACHER_CONTROL_TYPE:-add-plus}"
TEACHER_ARG="${TEACHER_ARG} --teacher_control_type ${TEACHER_CONTROL_TYPE}"

# ── --critic_ckpt (optional, for initializing critic from teacher weights) ──
CRITIC_CKPT_ARG=""
if [ -n "${CRITIC_CKPT:-}" ]; then
    CRITIC_CKPT_ARG="--critic_ckpt ${CRITIC_CKPT}"
    if [ "${RANK:-0}" = "0" ]; then
        echo "[runner] critic_ckpt: ${CRITIC_CKPT}"
    fi
fi

# ── --resume_from arg (optional, for resuming from a previous checkpoint) ──
RESUME_ARG=""
if [ -n "${RESUME_FROM:-}" ]; then
    RESUME_ARG="--resume_from ${RESUME_FROM}"
    if [ "${RANK}" = "0" ]; then
        echo "[launcher] [rank ${RANK}] resume from: ${RESUME_FROM}"
    fi
fi

# ── Run directory ──
# In multi-node mode all ranks share the same OUTPUT_DIR (it lives on shared
# storage that every node can see). RUN_TAG defaults to a hostname-derived
# value, but in multi-node setups different hostnames would cause each rank
# to write to a different directory. The most robust approach is to have the
# scheduler/platform pass RUN_TAG via env (or use a value derived from the
# master rank). By default we generate it from MASTER_PORT, which the
# scheduler assigns identically across all ranks.
export OUTPUT_DIR_BASE="${OUTPUT_DIR_BASE:-outputs}"

# ── Run naming ──
#
# Hierarchy:
#   RUN_NAME (user-friendly, optional, default 'run')
#     ↓
#   RUN_TAG = RUN_NAME + unique suffix (auto)
#     ↓
#   OUTPUT_DIR = OUTPUT_DIR_BASE/RUN_TAG
#
# The unique suffix avoids overwriting previous runs:
#   - Multi-node: use MASTER_PORT (platform-assigned, identical across all ranks).
#   - Single-node: use timestamp + PID.
#
# Multi-node MUST use a value all ranks agree on. $(date) at script eval time
# differs by 1-2s between nodes → different RUN_TAG per rank → ckpt save races
# and silent breakage. MASTER_PORT is platform-assigned and identical on every
# node, so it is safe to use.
#
# Examples:
#   RUN_NAME=mse_v1 bash scripts/train_mse.sh
#     → outputs/mse_v1_29500/ (multi-node)
#     → outputs/mse_v1_20260624_083012_pid1234/ (single-node)
#   bash scripts/train_mse.sh
#     → outputs/run_29500/ (default name 'run')
export RUN_NAME="${RUN_NAME:-run}"
if [ -n "${MASTER_PORT:-}" ]; then
    _RUN_SUFFIX="${MASTER_PORT}"
else
    _RUN_SUFFIX="$(date -u +%Y%m%d_%H%M%S)_pid$$"
fi
export RUN_TAG="${RUN_TAG:-${RUN_NAME}_${_RUN_SUFFIX}}"
export OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_DIR_BASE}/${RUN_TAG}}"
# Only rank 0 of the master machine creates dirs (avoid race)
if [ "${RANK}" = "0" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

# ── Hyperparameters ──
export LEARNING_RATE_GEN="${LEARNING_RATE_GEN:-1e-5}"
export LEARNING_RATE_CRITIC="${LEARNING_RATE_CRITIC:-4e-7}"   # unused in MSE-only stage
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
export MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
export WARMUP_RATIO="${WARMUP_RATIO:-0.0}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"

# ── DeepSpeed ZeRO config selection ──
# DS_ZERO_STAGE=3: configs_zero/zero3.yaml — full param sharding.
#                  Requires high-bandwidth interconnect (InfiniBand). Over
#                  plain TCP the cross-node all-gather traffic dominates.
# DS_ZERO_STAGE=2 (default for streaming): configs_zero/zero2_offload.yaml —
#                  params replicated (~10 GB per rank for a 5B model); only
#                  grad/opt-state are sharded. ~60x less cross-node traffic,
#                  works even without InfiniBand.
# Both stages support EMA.
export DS_ZERO_STAGE="${DS_ZERO_STAGE:-2}"
case "$DS_ZERO_STAGE" in
    2) export DS_CONFIG_FILE="${DS_CONFIG_FILE:-configs_zero/zero2_offload.yaml}" ;;
    3) export DS_CONFIG_FILE="${DS_CONFIG_FILE:-configs_zero/zero3.yaml}" ;;
    *) echo "ERROR: DS_ZERO_STAGE must be 2 or 3 (got: $DS_ZERO_STAGE)" >&2; exit 1 ;;
esac

# ── Generator EMA ──
# Works with both ZeRO-2 and ZeRO-3 (stage-aware dispatcher in train script).
export EMA_DECAY="${EMA_DECAY:-0.999}"
export EMA_START_STEP="${EMA_START_STEP:-200}"

# ── MSE σ schedule ──
export FLOW_SHIFT="${FLOW_SHIFT:-5.0}"

# ── Training steps ──
export NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-5000}"
# MSE_END_STEP: DMD activates at step > MSE_END_STEP. When equal to
# NUM_TRAIN_STEPS (default), the run stays in MSE mode start-to-finish.
export MSE_END_STEP="${MSE_END_STEP:-${NUM_TRAIN_STEPS}}"
# Internal CLI still uses the historical --ode_warmup_steps name; translate.
export ODE_WARMUP_STEPS="${MSE_END_STEP}"
export SAVE_STEPS="${SAVE_STEPS:-500}"
export LOGGING_STEPS="${LOGGING_STEPS:-1}"
export SEED="${SEED:-42}"

# ── Architecture ──
export NUM_LATENT_FRAMES="${NUM_LATENT_FRAMES:-21}"
export MAX_CACHE_FRAMES="${MAX_CACHE_FRAMES:-6}"
export PE_MODE="${PE_MODE:-slot}"
# Mixed 7F+21F support. Default=1 (21F-only, backward-compatible).
# Set FILTER_21F_ONLY=0 + NUM_LATENT_FRAMES=0 + a mixed-frame DATA_PATH to
# train on samples of varying frame lengths. Requires ZeRO-2.
export FILTER_21F_ONLY="${FILTER_21F_ONLY:-1}"

# ── DMD recipe knobs ──
# These are env-overridable. MSE-only runs (MSE_END_STEP=NUM_TRAIN_STEPS)
# silently no-op all of these via the DMD-branch guard inside the train
# script.
export NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-1}"
export STOCHASTIC_GRAD_TRUNCATION="${STOCHASTIC_GRAD_TRUNCATION:-0}"  # required when NUM_INFERENCE_STEPS>1
export DFAKE_GEN_UPDATE_RATIO="${DFAKE_GEN_UPDATE_RATIO:-1}"
export REAL_GUIDANCE_SCALE="${REAL_GUIDANCE_SCALE:-0.0}"
export RESET_LR_AT_DMD="${RESET_LR_AT_DMD:-0}"
# Optional rich-text negative prompt. Default = "" (legacy behavior, fine
# when REAL_GUIDANCE_SCALE=0). When REAL_GUIDANCE_SCALE>0, set this to a
# rich negative prompt to prevent CFG from amplifying the conditional bias
# (e.g. over-saturated colors).
export NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"

# ── Data ──
export DATA_PATH="${DATA_PATH:-/path/to/dataset.json}"
export MAX_SAMPLES="${MAX_SAMPLES:-0}"
if [ "$DATA_PATH" = "/path/to/dataset.json" ] || [ ! -f "$DATA_PATH" ]; then
    echo "ERROR: DATA_PATH=$DATA_PATH does not exist. Set DATA_PATH in config.yaml or environment." >&2
    exit 1
fi

# ── Cache-aware checkpointing ──
export STREAMING_GRAD_CHECKPOINT="${STREAMING_GRAD_CHECKPOINT:-1}"

# ── Decode (rank-0-only inside Python; multi-node still works) ──
export DECODE_EVERY="${DECODE_EVERY:-40}"
export DECODE_NUM_SCENES="${DECODE_NUM_SCENES:-4}"
export DECODE_MAX_FRAMES="${DECODE_MAX_FRAMES:-21}"
export DECODE_DTYPE="${DECODE_DTYPE:-fp32}"
export DECODE_WITH_TEACHER="${DECODE_WITH_TEACHER:-1}"
export DECODE_GRID_REPEATS="${DECODE_GRID_REPEATS:-3}"

# NCCL config + IB self-check is at the top of this script.

# ── Per-node multi-node sanity banner (all ranks print, so each node's log shows
#    its own view; for grep-ability use the [MULTINODE-PRELAUNCH] prefix) ──
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "[MULTINODE-PRELAUNCH] host=$(hostname) | This node's view of cluster:"
echo "[MULTINODE-PRELAUNCH]   WORLD_SIZE (machines)     = $WORLD_SIZE"
echo "[MULTINODE-PRELAUNCH]   RANK (this machine rank)  = $RANK"
echo "[MULTINODE-PRELAUNCH]   NPROC_PER_NODE            = $NPROC_PER_NODE"
echo "[MULTINODE-PRELAUNCH]   TOTAL_GPUS = WS×NPROC     = $TOTAL_GPUS"
echo "[MULTINODE-PRELAUNCH]   MASTER_ADDR               = $MASTER_ADDR"
echo "[MULTINODE-PRELAUNCH]   MASTER_PORT               = $MASTER_PORT"
echo "[MULTINODE-PRELAUNCH]   NCCL_SOCKET_IFNAME        = ${NCCL_SOCKET_IFNAME:-(unset)}"
echo "[MULTINODE-PRELAUNCH]   accelerate config (after envsubst) → see [MULTINODE-CHECK] lines in train script log"
echo "[MULTINODE-PRELAUNCH] After launch, verify in log:"
echo "[MULTINODE-PRELAUNCH]   grep '\\[MULTINODE-CHECK\\]' train.log | grep VERDICT"
echo "[MULTINODE-PRELAUNCH]   should see OK on every node, and actual_world_size=$TOTAL_GPUS on every rank"
echo "═══════════════════════════════════════════════════════════════════════════════"

# ── Banner (rank 0 only, to avoid log spam) ──
if [ "${RANK}" = "0" ]; then
    echo "═══════════════════════════════════════════════════════════════════════════════"
    echo "  TRAINING LAUNCH SUMMARY"
    echo
    echo "  ── Multi-node config ──"
    echo "  WORLD_SIZE (machines):       $WORLD_SIZE"
    echo "  RANK (this machine rank):    $RANK"
    echo "  NPROC_PER_NODE (GPUs/node):  $NPROC_PER_NODE"
    echo "  TOTAL_GPUS:                  $TOTAL_GPUS"
    echo "  MASTER_ADDR:                 $MASTER_ADDR"
    echo "  MASTER_PORT:                 $MASTER_PORT"
    echo "  NCCL_SOCKET_IFNAME:          ${NCCL_SOCKET_IFNAME:-(none)}"
    echo
    echo "  Base model:       $MODEL_PATH"
    echo "  Teacher ckpt:     ${TEACHER_CKPT}  (use_ema=${USE_EMA_TEACHER})"
    echo "  Data:             $DATA_PATH"
    echo "  Output:           $OUTPUT_DIR"
    echo
    echo "  ── Hyperparameters ──"
    echo "  lr_gen:           $LEARNING_RATE_GEN"
    echo "  GAS:              $GRADIENT_ACCUMULATION_STEPS  (effective batch = $TOTAL_GPUS × $GRADIENT_ACCUMULATION_STEPS = $((TOTAL_GPUS * GRADIENT_ACCUMULATION_STEPS)))"
    echo "  clip:             $MAX_GRAD_NORM"
    echo
    echo "  ── Architecture ──"
    echo "  num_latent_frames: $NUM_LATENT_FRAMES  (0=native length per sample)"
    echo "  filter_21f_only:   $FILTER_21F_ONLY  (0=mixed 7F+21F, 1=21F-only)"
    echo "  max_cache_frames:  $MAX_CACHE_FRAMES   (1 sink + $((MAX_CACHE_FRAMES - 1)) rolling)"
    echo "  pe_mode:           $PE_MODE"
    echo
    echo "  ── Schedule ──"
    echo "  mse_end_step:      $MSE_END_STEP  (DMD activates at step > this; == num_train_steps → DMD never starts)"
    echo "  ode_target_kind:   v_flow (single-step v-space MSE)"
    echo "  num_train_steps:   $NUM_TRAIN_STEPS"
    echo "═══════════════════════════════════════════════════════════════════════════════"
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export LAUNCH_TAG="${LAUNCH_TAG:-$(date -u +%Y%m%d_%H%M%S)_pid$$}"
echo "[launch] [rank ${RANK}] LAUNCH_TAG=$LAUNCH_TAG"

# ── Choose accelerate config (multi-node template) ──
# We must use configs_acc/streaming_multinode.yaml; the single-node yaml
# hard-codes num_machines/num_processes/machine_rank to 1/8/0. While the CLI
# flags --num_machines / --machine_rank / --num_processes look like they
# should override the yaml, accelerate's DeepSpeed config_file path actually
# prefers the yaml's hard-coded values. The result is that two machines
# silently start independent local jobs (no cross-node communication, ckpts
# overwrite each other).
#
# The multi-node template uses ${WORLD_SIZE}/${RANK}/${MASTER_ADDR}/
# ${MASTER_PORT}/${NPROC_PER_NODE} placeholders so each launch resolves to
# the actual cluster topology.
#
# accelerate parses the yaml with yaml.safe_load and does NOT expand
# ${ENV_VAR}, so we render the template into a resolved file before passing
# it to accelerate launch.
#
# We use plain sed (not envsubst) because envsubst comes from gettext-base
# which is not always installed on cluster images, whereas sed is universal.
ACCELERATE_CONFIG_TEMPLATE="${ACCELERATE_CONFIG_TEMPLATE:-${REPO_DIR}/configs_acc/streaming_multinode.yaml}"
ACCELERATE_CONFIG_RESOLVED="${OUTPUT_DIR}/accelerate_config_multinode.resolved_rank${RANK}.yaml"
if [ ! -f "$ACCELERATE_CONFIG_TEMPLATE" ]; then
    echo "ERROR: accelerate config template not found: $ACCELERATE_CONFIG_TEMPLATE" >&2
    exit 1
fi
mkdir -p "$(dirname "$ACCELERATE_CONFIG_RESOLVED")"
# Replace the placeholders with sed. Notes:
#  - use | as the sed delimiter to avoid clashing with / in IPs
#  - escape \$ so sed sees a literal $ (not a bash expansion)
sed -e "s|\${TOTAL_GPUS}|${TOTAL_GPUS}|g" \
    -e "s|\${WORLD_SIZE}|${WORLD_SIZE}|g" \
    -e "s|\${RANK}|${RANK}|g" \
    -e "s|\${MASTER_ADDR}|${MASTER_ADDR}|g" \
    -e "s|\${MASTER_PORT}|${MASTER_PORT}|g" \
    -e "s|\${NPROC_PER_NODE}|${NPROC_PER_NODE}|g" \
    < "$ACCELERATE_CONFIG_TEMPLATE" > "$ACCELERATE_CONFIG_RESOLVED"

# Verify that no ${VAR} placeholders remain on non-comment lines (any
# left-over placeholder = missed sed substitution = yaml.safe_load will fail).
# Comment lines (with optional leading whitespace) are documentation and
# don't count as missed substitutions.
if grep -vE '^\s*#' "$ACCELERATE_CONFIG_RESOLVED" | grep -E '\$\{[A-Z_]+\}' > /dev/null; then
    echo "ERROR: resolved yaml still has unsubstituted \${VAR} on non-comment lines:" >&2
    grep -vE '^\s*#' "$ACCELERATE_CONFIG_RESOLVED" | grep -nE '\$\{[A-Z_]+\}' >&2
    echo "  → ensure every placeholder used in the template is also in the sed pipeline above" >&2
    exit 1
fi

if [ "${RANK}" = "0" ]; then
    echo "[launcher] [rank ${RANK}] resolved accelerate yaml → $ACCELERATE_CONFIG_RESOLVED"
    echo "─── resolved yaml content ───"
    cat "$ACCELERATE_CONFIG_RESOLVED"
    echo "─────────────────────────────"
fi
ACCELERATE_CONFIG_FILE="$ACCELERATE_CONFIG_RESOLVED"

# ── Tee log: rank 0 → train.log (main), other ranks → logs/rank{N}_<tag>.log ──
mkdir -p "${OUTPUT_DIR}/logs"
LOG_FILE="${OUTPUT_DIR}/logs/rank${RANK}_${LAUNCH_TAG}.log"
if [ "${RANK}" = "0" ]; then
    # rank 0 also writes train.log (kept for single-node compatibility)
    LOG_TARGET="$OUTPUT_DIR/train.log"
else
    LOG_TARGET="$LOG_FILE"
fi
mkdir -p "$(dirname "$LOG_TARGET")"

# ── Optional: install imageio-ffmpeg for mp4 video saving ──
# Saving decoded mp4 videos (DECODE_SAVE_VIDEOS=1) needs an ffmpeg backend.
# Some cluster images only ship PyAV (without an ffmpeg binary), and
# imageio's default API on top of PyAV breaks with kwargs/codec mismatches.
# imageio-ffmpeg is a small (~4 MB) pip package that bundles a static ffmpeg
# binary; once installed, imageio automatically prefers it. Idempotent
# (already-installed → no-op). Failures (no network / sandboxed) do NOT
# block training — video saving will simply be skipped.
if [ "${DECODE_SAVE_VIDEOS:-0}" = "1" ]; then
    pip install -q imageio-ffmpeg 2>&1 | tail -2 || echo "[wrapper] WARN: imageio-ffmpeg install failed; mp4 save will skip"
fi

# ── Launch ──
# All multi-node parameters are read from env and passed explicitly to
# accelerate launch.
# Wrap in a retry loop to recover from transient failures (e.g. OSS-FUSE
# mount drops, NCCL timeouts). Set MAX_RETRIES=0 to disable retries.
MAX_RETRIES="${MAX_RETRIES:-5}"
RETRY_DELAY="${RETRY_DELAY:-30}"
attempt=0
while :; do
    attempt=$((attempt + 1))
    echo "═══════════════════════════════════════════════════════════════════════════════"
    echo "[launch] Attempt $attempt of $((MAX_RETRIES + 1))"
    echo "═══════════════════════════════════════════════════════════════════════════════"
accelerate launch \
    --config_file "$ACCELERATE_CONFIG_FILE" \
    --num_processes "$TOTAL_GPUS" \
    --num_machines "$WORLD_SIZE" \
    --machine_rank "$RANK" \
    --main_process_ip "$MASTER_ADDR" \
    --main_process_port "$MASTER_PORT" \
    --mixed_precision bf16 \
    -m core.streaming.train_distill \
    --model_path "$MODEL_PATH" \
    --data_path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    $TEACHER_ARG \
    $RESUME_ARG \
    $CRITIC_CKPT_ARG \
    --use_control_dataset \
    --num_train_steps "$NUM_TRAIN_STEPS" \
    --batch_size 1 \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --learning_rate_gen "$LEARNING_RATE_GEN" \
    --learning_rate_critic "$LEARNING_RATE_CRITIC" \
    --weight_decay "$WEIGHT_DECAY" \
    --adam_beta1 0.0 \
    --adam_beta2 0.999 \
    --adam_eps 1e-8 \
    --num_latent_frames "$NUM_LATENT_FRAMES" \
    --filter_21f_only "$FILTER_21F_ONLY" \
    --num_inference_steps "$NUM_INFERENCE_STEPS" \
    --guidance_scale 1.0 \
    --real_guidance_scale "$REAL_GUIDANCE_SCALE" \
    --negative_prompt "$NEGATIVE_PROMPT" \
    --dfake_gen_update_ratio "$DFAKE_GEN_UPDATE_RATIO" \
    $([ "${STOCHASTIC_GRAD_TRUNCATION:-0}" = "1" ] && echo "--stochastic_grad_truncation") \
    $([ "${RESET_LR_AT_DMD:-0}" = "1" ] && echo "--reset_lr_at_dmd") \
    --max_grad_norm "$MAX_GRAD_NORM" \
    --bf16 \
    --logging_steps "$LOGGING_STEPS" \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit "${SAVE_TOTAL_LIMIT:-0}" \
    --max_samples "$MAX_SAMPLES" \
    --seed "$SEED" \
    --warmup_ratio "$WARMUP_RATIO" \
    --min_step_frac 0.02 \
    --max_step_frac 0.98 \
    --decode_every "$DECODE_EVERY" \
    --decode_dtype "$DECODE_DTYPE" \
    --decode_num_scenes "$DECODE_NUM_SCENES" \
    --decode_max_frames "$DECODE_MAX_FRAMES" \
    --decode_with_teacher \
    --max_cache_frames "$MAX_CACHE_FRAMES" \
    --pe_mode "$PE_MODE" \
    --ode_warmup_steps "$ODE_WARMUP_STEPS" \
    --ode_target_kind v_flow \
    --ema_decay "$EMA_DECAY" \
    --ema_start_step "$EMA_START_STEP" \
    2>&1 | tee "$LOG_TARGET"
    rc=${PIPESTATUS[0]}
    if [ "$rc" = "0" ]; then
        echo "[launch] Training completed successfully on attempt $attempt"
        break
    fi
    if [ "$attempt" -ge "$((MAX_RETRIES + 1))" ]; then
        echo "[launch] Exceeded max retries ($MAX_RETRIES). Giving up."
        exit "$rc"
    fi
    echo "[launch] Attempt $attempt failed (exit code $rc). Retrying in ${RETRY_DELAY}s..."
    sleep "$RETRY_DELAY"
done

}
exit $?
