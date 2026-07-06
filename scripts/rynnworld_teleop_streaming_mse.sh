#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# train_mse.sh — Stage 1: MSE pretraining entry point.
#
# Trains the student causal transformer with v-flow MSE loss on (control, video)
# pairs. The student learns to denoise from a control skeleton conditioned on
# a single image latent, in streaming mode (one frame at a time with KV cache).
#
# Paths come from configs/config.yaml (loaded by rynnworld_teleop_streaming_runner.sh); override per-run
# via env vars on the command line:
#   bash scripts/train_mse.sh                    # uses configs/config.yaml defaults
#   RUN_NAME=ablation_v1 bash scripts/train_mse.sh
#   NUM_TRAIN_STEPS=20000 bash scripts/train_mse.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Data mode ──
# Mixed 7F + 21F mode (set FILTER_21F_ONLY=1 to filter only 21-frame samples).
# NUM_LATENT_FRAMES=0 means use each sample's native frame count.
export FILTER_21F_ONLY="${FILTER_21F_ONLY:-0}"
export NUM_LATENT_FRAMES="${NUM_LATENT_FRAMES:-0}"

# ── Teacher control type (skeleton injection mode) ──
export TEACHER_CONTROL_TYPE="${TEACHER_CONTROL_TYPE:-add}"

# ── Run setup ──
export RUN_NAME="${RUN_NAME:-mse_sft}"
export OUTPUT_DIR_BASE="${OUTPUT_DIR_BASE:-outputs/mse}"

# ── Training schedule ──
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
export SAVE_STEPS="${SAVE_STEPS:-100}"
export NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-4000}"
# MSE-only run: keep MSE_END_STEP == NUM_TRAIN_STEPS so DMD never activates.
export MSE_END_STEP="${MSE_END_STEP:-4000}"
export SKIP_FIRST_DECODE="${SKIP_FIRST_DECODE:-1}"
export DS_ZERO_STAGE="${DS_ZERO_STAGE:-2}"
export DECODE_SAVE_VIDEOS="${DECODE_SAVE_VIDEOS:-1}"

# ── Block streaming + FixedSizeCache (Self-Forcing aligned) ──
# Forward N frames per block (bidirectional within, causal across).
# FixedSizeCache uses a pre-allocated KV buffer with in-place writes so that
# gradient_checkpointing stays compatible.
export USE_FIXED_CACHE="${USE_FIXED_CACHE:-1}"
export NUM_FRAME_PER_BLOCK="${NUM_FRAME_PER_BLOCK:-3}"
export NUM_MAX_FRAMES="${NUM_MAX_FRAMES:-21}"
export SINK_SIZE="${SINK_SIZE:-1}"
export LOCAL_ATTN_SIZE="${LOCAL_ATTN_SIZE:--1}"

# DATA_PATH, TEACHER_CKPT are loaded from configs/config.yaml by rynnworld_teleop_streaming_runner.sh.
exec bash "${SCRIPT_DIR}/rynnworld_teleop_streaming_runner.sh"
