#!/bin/bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export PATH="/mnt/workspace/umi-world-model-lab/projects/minimax_adr_generation/.venv/bin:$PATH"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export MASTER_ADDR=localhost
export MASTER_PORT="${MASTER_PORT:-29744}"

TRAIN_MANIFEST="${TRAIN_MANIFEST:-/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/agibot_action_v10_state_spatial_16_v1/train.json}"
OUTPUT_DIR="${OUTPUT_DIR:-training/native_action_v10_state_spatial_16_500}"
TRAIN_STEPS="${TRAIN_STEPS:-500}"

accelerate launch --config_file configs_acc/1gpu_smoke.yaml finetune.py \
  --model_path pretrained/Wan2.2-TI2V-5B-Diffusers \
  --model_name rynnworld_teleop --model_type rynnworld_teleop \
  --training_type lora --rank 8 --lora_alpha 8 \
  --target_modules attn1.to_q attn1.to_k attn1.to_v attn1.to_out.0 \
  --freeze_lora true \
  --condition_mode native_trajectory --native_conditioner_version v10 \
  --native_trajectory_dim 148 \
  --native_baseline_init reports/direct_action/gate_d/run_015/checkpoints/adapter_step1000.pt \
  --native_adapter_init training/native_action_v8_single_clip_zeroinit_300/checkpoint-300 \
  --action_dropout_prob 0.0 \
  --action_contrastive_weight 0.0 --action_ranking_weight 0.0 \
  --action_motion_weight 0.0 \
  --action_local_motion_weight 1.0 --action_local_motion_focus 4.0 \
  --action_spatial_gate_weight 0.0 \
  --init_from_checkpoint pretrained/RynnWorld-Teleop-Causal \
  --output_dir "$OUTPUT_DIR" \
  --report_to tensorboard --train_resolution 33x480x832 \
  --train_epochs 500 --train_steps "$TRAIN_STEPS" --seed 42 \
  --batch_size 1 --gradient_accumulation_steps 1 --mixed_precision bf16 \
  --num_workers 0 --pin_memory true \
  --checkpointing_steps 50 --checkpointing_limit 4 \
  --do_validation false \
  --validation_dir "$TRAIN_MANIFEST" \
  --cache_dir data --control_type add \
  --learning_rate 1e-4 --control_lr 1e-4 --native_projection_lr 1e-4 \
  --max_grad_norm 1.0 --lr_scheduler cosine --lr_warmup_steps 20 \
  --ema_decay 0.99 --ema_start_step 1000000 --prompt ''
