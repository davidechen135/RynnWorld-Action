#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# train_dmd.sh — Stage 2: DMD distillation entry point.
#
# Resumes from an MSE checkpoint and adversarially distills against the
# teacher. The student generates latents in a few denoising steps; the
# critic (jointly trained) and the frozen teacher together produce the DMD
# gradient that pushes the student toward the teacher's distribution.
#
# Required overrides:
#   RESUME_FROM        path to an MSE checkpoint directory (must contain
#                      generator.pt, critic.pt, control_running_stats.bin)
#   MSE_END_STEP       step count of that MSE checkpoint. DMD activates at
#                      step > MSE_END_STEP (no MSE re-run).
#
# Example:
#   RESUME_FROM=outputs/mse/<run>/checkpoint-4000 \
#     MSE_END_STEP=4000 \
#     bash scripts/rynnworld_teleop_streaming_dmd.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Resume from MSE checkpoint ──
# RESUME_FROM is REQUIRED; without it DMD starts from the base model.
# MSE_END_STEP marks the MSE→DMD boundary; DMD activates at step > MSE_END_STEP.
export MSE_END_STEP="${MSE_END_STEP:-4000}"

# ── Data mode ──
export FILTER_21F_ONLY="${FILTER_21F_ONLY:-0}"
export NUM_LATENT_FRAMES="${NUM_LATENT_FRAMES:-0}"

# ── Run setup ──
export RUN_NAME="${RUN_NAME:-dmd}"
export OUTPUT_DIR_BASE="${OUTPUT_DIR_BASE:-outputs/dmd}"
export NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-$((MSE_END_STEP + 3000))}"
export SAVE_STEPS="${SAVE_STEPS:-100}"

# ── Debug decode ──
export DECODE_EVERY="${DECODE_EVERY:-200}"
export DECODE_SAVE_VIDEOS="${DECODE_SAVE_VIDEOS:-1}"
export DECODE_VIDEO_ROW_WHITELIST="${DECODE_VIDEO_ROW_WHITELIST:-student,control}"
export DECODE_NUM_SCENES="${DECODE_NUM_SCENES:-10}"
export DECODE_FIXED_SAMPLE_IDS="${DECODE_FIXED_SAMPLE_IDS:-0,180000,360000,540000,720000}"
export DECODE_GRID_REPEATS="${DECODE_GRID_REPEATS:-1}"
export SKIP_FIRST_DECODE="${SKIP_FIRST_DECODE:-0}"

# ── Distributed training ──
# ZeRO-2 by default (ZeRO-3 requires libibverbs / InfiniBand).
# GAS=2 on 64 GPUs → effective batch = 128.
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
export DS_ZERO_STAGE="${DS_ZERO_STAGE:-2}"

# ── DMD: 4-step student rollout with stochastic gradient truncation ──
# SGT retains gradients for one randomly-chosen denoising step per iteration,
# reducing activation memory ~4×.
export NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-4}"
export STOCHASTIC_GRAD_TRUNCATION="${STOCHASTIC_GRAD_TRUNCATION:-1}"

# ── Critic update schedule ──
export DFAKE_GEN_UPDATE_RATIO="${DFAKE_GEN_UPDATE_RATIO:-5}"
export LEARNING_RATE_CRITIC="${LEARNING_RATE_CRITIC:-5e-7}"

# ── Optimization ──
# No LR warmup: matches CausVid / Self-Forcing recipe.
export MAX_GRAD_NORM="${MAX_GRAD_NORM:-1.0}"
export WARMUP_RATIO="${WARMUP_RATIO:-0.0}"

# ── Real-score CFG scale ──
# Set to 0.0 when the base model is already CFG-distilled — nonzero guidance
# amplifies latent magnitude during DMD and can trigger mode collapse.
# CFG=0 also skips the uncond forward, saving wall time.
export REAL_GUIDANCE_SCALE="${REAL_GUIDANCE_SCALE:-0.0}"
export NEGATIVE_PROMPT="${NEGATIVE_PROMPT:-}"

# ── Block-level streaming (Self-Forcing: num_frame_per_block=3) ──
# Forward N frames together per block: bidirectional within block, causal
# across blocks (KV cache). Set to 1 to fall back to per-frame streaming.
export NUM_FRAME_PER_BLOCK="${NUM_FRAME_PER_BLOCK:-3}"

# ── FixedSizeCache (replaces DynamicCache for block streaming) ──
# Pre-allocated KV cache + in-place writes → compatible with
# gradient_checkpointing. Requires NUM_FRAME_PER_BLOCK > 1.
export USE_FIXED_CACHE="${USE_FIXED_CACHE:-1}"
export NUM_MAX_FRAMES="${NUM_MAX_FRAMES:-21}"   # cache buffer size in frames
export SINK_SIZE="${SINK_SIZE:-1}"               # keep frame 0 (condition)
export LOCAL_ATTN_SIZE="${LOCAL_ATTN_SIZE:--1}"  # -1 = full cache, no eviction

exec bash "${SCRIPT_DIR}/rynnworld_teleop_streaming_runner_dmd.sh"
