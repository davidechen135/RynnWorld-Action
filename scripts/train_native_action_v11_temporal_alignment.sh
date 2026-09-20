#!/bin/bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export PATH="/mnt/workspace/umi-world-model-lab/projects/minimax_adr_generation/.venv/bin:$PATH"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export MASTER_ADDR=localhost
export MASTER_PORT="${MASTER_PORT:-29746}"

TRAIN_MANIFEST="${TRAIN_MANIFEST:-/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/agibot_action_v10_state_256_v1/train.json}"
OUTPUT_DIR="${OUTPUT_DIR:-training/native_action_v11_temporal_alignment_300}"
TRAIN_STEPS="${TRAIN_STEPS:-300}"

accelerate launch --config_file configs_acc/1gpu_smoke.yaml finetune.py \
  --model_path pretrained/Wan2.2-TI2V-5B-Diffusers \
  --model_name rynnworld_teleop --model_type rynnworld_teleop \
  --training_type lora --rank 8 --lora_alpha 8 \
  --target_modules attn1.to_q attn1.to_k attn1.to_v attn1.to_out.0 \
  --freeze_lora true \
  --condition_mode native_trajectory --native_conditioner_version v11 \
  --native_trajectory_dim 148 \
  --native_baseline_init reports/direct_action/gate_d/run_015/checkpoints/adapter_step1000.pt \
  --native_adapter_init training/native_action_v10b_state_256_600/checkpoint-400 \
  --action_dropout_prob 0.1 \
  --action_contrastive_weight 0.0 --action_ranking_weight 0.0 \
  --action_motion_weight 1.0 \
  --action_local_motion_weight 0.5 --action_local_motion_focus 2.0 \
  --action_spatial_gate_weight 0.0 \
  --init_from_checkpoint pretrained/RynnWorld-Teleop-Causal \
  --output_dir "$OUTPUT_DIR" \
  --report_to tensorboard --train_resolution 33x480x832 \
  --train_epochs 10 --train_steps "$TRAIN_STEPS" --seed 42 \
  --batch_size 1 --gradient_accumulation_steps 1 --mixed_precision bf16 \
  --num_workers 0 --pin_memory true \
  --checkpointing_steps 150 --checkpointing_limit 3 \
  --do_validation false \
  --validation_dir "$TRAIN_MANIFEST" \
  --cache_dir data --control_type add \
  --learning_rate 2e-5 --control_lr 2e-5 --native_projection_lr 1e-5 \
  --max_grad_norm 1.0 --lr_scheduler cosine --lr_warmup_steps 20 \
  --ema_decay 0.99 --ema_start_step 1000000 --prompt ''
