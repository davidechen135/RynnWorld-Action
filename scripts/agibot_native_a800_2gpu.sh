#!/bin/bash
set -euo pipefail

export TOKENIZERS_PARALLELISM=false
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export MASTER_ADDR="${MASTER_ADDR:-localhost}"
export MASTER_PORT="${MASTER_PORT:-29613}"
export MODEL_PATH="${MODEL_PATH:-pretrained/Wan2.2-TI2V-5B-Diffusers}"
export DATA_JSON="${DATA_JSON:-data/agibot_362_native_v1/train.json}"
export OUTPUT_DIR="${OUTPUT_DIR:-training/agibot_362_native_f2}"
export TRAIN_STEPS="${TRAIN_STEPS:-200}"
export GRAD_ACCUM="${GRAD_ACCUM:-4}"
export CONTROL_LR="${CONTROL_LR:-5e-5}"
export BASE_LR="${BASE_LR:-5e-6}"
export FREEZE_LORA="${FREEZE_LORA:-true}"

accelerate launch --config_file configs_acc/2gpu_a800_native.yaml \
  --main_process_ip "$MASTER_ADDR" --main_process_port "$MASTER_PORT" \
  --num_processes 2 finetune.py \
  --model_path "$MODEL_PATH" --model_name rynnworld_teleop \
  --model_type rynnworld_teleop --training_type lora \
  --rank 16 --lora_alpha 16 \
  --target_modules attn1.to_q attn1.to_k attn1.to_v attn1.to_out.0 \
  --freeze_lora "$FREEZE_LORA" --condition_mode native_trajectory \
  --init_from_checkpoint pretrained/RynnWorld-Teleop \
  --output_dir "$OUTPUT_DIR" --report_to tensorboard \
  --train_resolution 81x480x832 --train_steps "$TRAIN_STEPS" \
  --train_epochs 100 --seed 42 --batch_size 1 \
  --gradient_accumulation_steps "$GRAD_ACCUM" --mixed_precision bf16 \
  --num_workers 4 --pin_memory true --nccl_timeout 7200 \
  --checkpointing_steps 100 --checkpointing_limit 3 \
  --do_validation false --validation_dir "$DATA_JSON" --cache_dir data \
  --control_type add --learning_rate "$BASE_LR" --control_lr "$CONTROL_LR" \
  --max_grad_norm 1.0 --beta2 0.999 --weight_decay 0.01 --epsilon 1e-10 \
  --lr_scheduler cosine --lr_warmup_steps 5 \
  --ema_decay 0.99 --ema_start_step 1000000 --prompt ""
