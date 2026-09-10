#!/bin/bash
# Gate D Phase 5/7 smoke test: single-GPU forward/backward check on the new
# fixed-33-frame native_trajectory data (/mnt/workspace/umi-world-model-lab/datasets/rynnworld-teleop/agibot_action_core_33f_v1).
# Goal: verify the native_trajectory_dim=37 path runs end-to-end (Dataset load,
# DataLoader/collate, compute_loss, backward) without NaN/OOM. NOT for convergence.
set -euo pipefail
export TOKENIZERS_PARALLELISM=false
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export MASTER_ADDR=localhost
export MASTER_PORT=29711
export MODEL_PATH="${MODEL_PATH:-pretrained/Wan2.2-TI2V-5B-Diffusers}"
export DATA_JSON="${DATA_JSON:-data/agibot_action_core_33f_v1/train.json}"
export OUTPUT_DIR="${OUTPUT_DIR:-training/gate_d_33f_smoke}"
export TRAIN_STEPS="${TRAIN_STEPS:-3}"

accelerate launch --config_file configs_acc/1gpu_smoke.yaml finetune.py \
  --model_path "$MODEL_PATH" \
  --model_name rynnworld_teleop \
  --model_type rynnworld_teleop \
  --training_type lora \
  --rank 16 --lora_alpha 16 \
  --target_modules attn1.to_q attn1.to_k attn1.to_v attn1.to_out.0 \
  --freeze_lora true \
  --condition_mode native_trajectory \
  --native_trajectory_dim 37 \
  --init_from_checkpoint pretrained/RynnWorld-Teleop \
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
  --nccl_timeout 1800 \
  --checkpointing_steps 1000 \
  --checkpointing_limit 1 \
  --do_validation false \
  --validation_dir "$DATA_JSON" \
  --cache_dir data \
  --control_type add \
  --learning_rate 5e-4 \
  --control_lr 5e-4 \
  --max_grad_norm 1.0 \
  --beta2 0.999 \
  --weight_decay 0.01 \
  --epsilon 1e-10 \
  --lr_scheduler cosine \
  --lr_warmup_steps 1 \
  --ema_decay 0.99 \
  --ema_start_step 1000000 \
  --prompt ''
