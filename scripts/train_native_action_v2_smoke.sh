#!/bin/bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export PATH="/mnt/workspace/umi-world-model-lab/projects/minimax_adr_generation/.venv/bin:$PATH"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export MASTER_ADDR=localhost
export MASTER_PORT="${MASTER_PORT:-29721}"
export OUTPUT_DIR="${OUTPUT_DIR:-training/native_action_v2_raw33_contrastive_fixed_smoke}"
export TRAIN_STEPS="${TRAIN_STEPS:-150}"

accelerate launch --config_file configs_acc/1gpu_smoke.yaml finetune.py \
  --model_path pretrained/Wan2.2-TI2V-5B-Diffusers \
  --model_name rynnworld_teleop \
  --model_type rynnworld_teleop \
  --training_type lora \
  --rank 8 --lora_alpha 8 \
  --target_modules attn1.to_q attn1.to_k attn1.to_v attn1.to_out.0 \
  --freeze_lora false \
  --condition_mode native_trajectory \
  --native_conditioner_version v2 \
  --native_trajectory_dim 33 \
  --action_dropout_prob 0.15 \
  --action_contrastive_weight 1.0 \
  --action_contrastive_margin 0.02 \
  --init_from_checkpoint pretrained/RynnWorld-Teleop-Causal \
  --output_dir "$OUTPUT_DIR" \
  --report_to tensorboard \
  --train_resolution 33x480x832 \
  --train_epochs 100 \
  --train_steps "$TRAIN_STEPS" \
  --seed 42 \
  --batch_size 1 \
  --gradient_accumulation_steps 1 \
  --mixed_precision bf16 \
  --num_workers 0 \
  --pin_memory true \
  --checkpointing_steps 50 \
  --checkpointing_limit 4 \
  --do_validation false \
  --validation_dir /mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/agibot_action_raw33_v2_smoke/train.json \
  --cache_dir data \
  --control_type add \
  --learning_rate 1e-4 \
  --control_lr 5e-4 \
  --max_grad_norm 1.0 \
  --lr_scheduler cosine \
  --lr_warmup_steps 20 \
  --ema_decay 0.99 \
  --ema_start_step 1000000 \
  --prompt ''
