export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-storage_bond}"
export NCCL_CROSS_NIC=1
export NCCL_IB_TIMEOUT=22

set -ex

echo "START TIME: $(date)"
echo "Running on host: $(hostname)"

if [ -z "$WORLD_SIZE" ] || [ -z "$RANK" ] || [ -z "$MASTER_ADDR" ] || [ -z "$MASTER_PORT" ] || [ -z "$NPROC_PER_NODE" ]; then
    echo "CRITICAL ERROR: Platform did not set required env vars for torchrun."
    exit 1
fi

TOTAL_GPUS=$((WORLD_SIZE * NPROC_PER_NODE))

echo "--- Accelerate Launch Arguments ---"
echo "Total Processes (TOTAL_GPUS): $TOTAL_GPUS"
echo "Total Machines (NNODES): $WORLD_SIZE"
echo "Current Machine Rank (NODE_RANK): $RANK"
echo "MASTER: $MASTER_ADDR : $MASTER_PORT"
echo "-----------------------------------"

PROGRAM_FILE="finetune.py"
MODEL_ARGS=(
    --model_path "${MODEL_PATH:-pretrained/Wan2.2-TI2V-5B-Diffusers}"
    --model_name rynnworld_teleop_pretrain
    --model_type rynnworld_teleop_pretrain
)

OUTPUT_DIR="training/rynnworld-teleop-${TOTAL_GPUS}gpu-SFT-pretrain"

TRAINING_ARGS=(
    --training_type sft
    --train_epochs 1 --seed 42 --batch_size 1
    --gradient_accumulation_steps 16 --mixed_precision bf16
    --num_workers 8 --pin_memory True --nccl_timeout 7200
)

OPTIMIZER_ARGS=(
    --learning_rate 2e-5 --max_grad_norm 1.0
    --beta2 0.999 --weight_decay 0.01 --epsilon 1e-10
    --lr_scheduler cosine --lr_warmup_steps 200
    --ema_decay 0.999 --ema_start_step 300
    --prompt ''
)

LOG_ARGS=(
    --output_dir $OUTPUT_DIR --report_to tensorboard
    --checkpointing_steps 100 --checkpointing_limit 20 --resume_from_checkpoint "latest"
)

DATA_ARGS=(
    --train_resolution 81x480x832 --do_validation false
    --validation_dir data/sample_data.json 
    --cache_dir data
)

export TOKENIZERS_PARALLELISM=false
export ACCELERATE_CONFIG_FILE="configs_acc/multinode_deepspeed.yaml"

accelerate launch \
    --config_file "$ACCELERATE_CONFIG_FILE" \
    --num_processes "$TOTAL_GPUS" \
    --num_machines "$WORLD_SIZE" \
    --machine_rank "$RANK" \
    --main_process_ip "$MASTER_ADDR" \
    --main_process_port "$MASTER_PORT" \
    $PROGRAM_FILE \
    "${MODEL_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${LOG_ARGS[@]}" \
    "${DATA_ARGS[@]}"

echo "END TIME: $(date)"
