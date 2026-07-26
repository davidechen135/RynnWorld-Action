#!/bin/bash
# Minimal 2-GPU SFT training smoke test on 3 shared clips (data/sample_data.json).
# Goal: verify the training pipeline runs on dual H20 without OOM/errors, capture
# loss + VRAM peak + step speed. NOT for convergence.
set -e
export TOKENIZERS_PARALLELISM=false
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export MASTER_ADDR=localhost
export MASTER_PORT=29511
export NNODES=1
export NUM_PROCESSES=2
export MODEL_PATH="${MODEL_PATH:-pretrained/Wan2.2-TI2V-5B-Diffusers}"

LAUNCHER="accelerate launch \
    --config_file configs_acc/2gpu.yaml \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank 0 \
    --num_processes $NUM_PROCESSES \
    --num_machines $NNODES"

PROGRAM="finetune.py \
    --model_path $MODEL_PATH \
    --model_name rynnworld_teleop \
    --model_type rynnworld_teleop \
    --training_type sft \
    --output_dir training/smoke_sft_2gpu \
    --report_to tensorboard \
    --train_resolution 81x480x832 \
    --train_epochs 1 \
    --train_steps 20 \
    --seed 42 \
    --batch_size 1 \
    --gradient_accumulation_steps 1 \
    --mixed_precision bf16 \
    --num_workers 2 \
    --pin_memory True \
    --nccl_timeout 7200 \
    --checkpointing_steps 1000 \
    --checkpointing_limit 1 \
    --do_validation false \
    --validation_dir data/sample_data.json \
    --cache_dir data \
    --control_type add \
    --learning_rate 2e-5 \
    --control_lr 2e-5 \
    --max_grad_norm 1.0 \
    --beta2 0.999 \
    --weight_decay 0.01 \
    --epsilon 1e-10 \
    --lr_scheduler cosine \
    --lr_warmup_steps 200 \
    --ema_decay 0.999 \
    --ema_start_step 100000 \
    --prompt ''"

eval "$LAUNCHER $PROGRAM"
