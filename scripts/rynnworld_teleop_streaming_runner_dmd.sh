#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# train_runner_dmd.sh — DMD-specific environment defaults.
#
# Sets DMD-phase defaults (different from MSE) then hands off to the main
# launcher (rynnworld_teleop_streaming_runner.sh) for the actual accelerate launch.
#
# Differences vs MSE phase:
#   - Teacher must be a full pretrained checkpoint with control_patch_embedding
#     (no base init step).
#   - LR/grad-clip/warmup tuned for adversarial (DMD) training.
#   - EMA decay 0.99 (LongLive recipe).
#   - σ schedule shifted to LongLive's flow_shift=5.0 (concentrates timesteps
#     in the high-noise regime where DMD is most informative).
#
# Usage: see scripts/rynnworld_teleop_streaming_dmd.sh (which sets
# RESUME_FROM/MSE_END_STEP then execs into this).
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Teacher (full pretrained Wan2.2 + control_patch_embedding) ──
# TEACHER_CKPT is loaded from configs/config.yaml by rynnworld_teleop_streaming_runner.sh.
export TEACHER_INIT_FROM_CHECKPOINT="${TEACHER_INIT_FROM_CHECKPOINT:-}"
export TEACHER_CONTROL_TYPE="${TEACHER_CONTROL_TYPE:-add}"
export USE_EMA_TEACHER="${USE_EMA_TEACHER:-0}"

# ── DMD optimization hyperparameters ──
export LEARNING_RATE_GEN="${LEARNING_RATE_GEN:-2e-6}"
export LEARNING_RATE_CRITIC="${LEARNING_RATE_CRITIC:-5e-7}"
export MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
export WARMUP_RATIO="${WARMUP_RATIO:-0.0}"
export DFAKE_GEN_UPDATE_RATIO="${DFAKE_GEN_UPDATE_RATIO:-5}"
export RESET_LR_AT_DMD="${RESET_LR_AT_DMD:-0}"

# ── EMA (LongLive recipe) ──
export EMA_DECAY="${EMA_DECAY:-0.99}"
# EMA starts 200 steps after the MSE→DMD boundary, giving the student a brief
# window to settle before tracking begins.
if [ -z "${EMA_START_STEP:-}" ]; then
    if [ -n "${MSE_END_STEP:-}" ]; then
        export EMA_START_STEP=$((MSE_END_STEP + 200))
    else
        export EMA_START_STEP=200
    fi
fi

# ── DMD σ schedule ──
# flow_shift=5.0 concentrates timesteps in [0.5, 1.0] (high-noise band) where
# the teacher's denoising signal is most informative. See LongLive paper.
export FLOW_SHIFT="${FLOW_SHIFT:-5.0}"
export INFERENCE_FLOW_SHIFT="${INFERENCE_FLOW_SHIFT:-5.0}"
export DMD_FLOW_SHIFT="${DMD_FLOW_SHIFT:-5.0}"
# Single timestep per video clip (rather than per-frame), reducing within-batch
# variance of the DMD gradient.
export DMD_UNIFORM_TIMESTEP="${DMD_UNIFORM_TIMESTEP:-1}"

# ── Resume guard ──
# Without RESUME_FROM, DMD would start from base Wan2.2 + teacher CPE and
# spend ~8000 steps re-learning what MSE already taught. Strongly recommend
# setting RESUME_FROM to an MSE checkpoint.
if [ -z "${RESUME_FROM:-}" ]; then
    echo "═══════════════════════════════════════════════════════════════════════════════"
    echo "  WARNING: RESUME_FROM not set — DMD will start from base Wan2.2."
    echo "  Recommended: set RESUME_FROM=<path to an MSE checkpoint directory>"
    echo "  e.g.: outputs/mse/run_<port>/checkpoint-N"
    echo "═══════════════════════════════════════════════════════════════════════════════"
fi

exec bash "${SCRIPT_DIR}/rynnworld_teleop_streaming_runner.sh"
