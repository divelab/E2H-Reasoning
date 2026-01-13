#!/bin/bash

# Training script for variance regularized curriculum scheduler
# This script trains a model using the new OOD generalization-based scheduler

# Check GPU availability first
echo "Checking GPU availability..."
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

# Set environment variables
export WANDB_PROJECT=Sys2Bench-VarReg
export ROOT_PATH=/data/shurui.gui/Projects/Sys2Bench

# Training configuration
GPUS="0,1"  # Change based on available GPUs
MODEL="qwen15"  # qwen15 for 1.5B, qwen for 3B
TASK="countdown6"
SCHEDULER="variance_regularized"
MAX_STEPS=1600
BATCH_SIZE=2
PORT=29850

echo "Starting training with variance regularized scheduler..."
echo "Model: $MODEL"
echo "Task: $TASK"
echo "GPUs: $GPUS"

# Run training
CUDA_VISIBLE_DEVICES=$GPUS accelerate launch \
    --num_processes 1 \
    --main_process_port=$PORT \
    --config_file methods/RL/deep_speed.yaml \
    methods/RL/main.py \
    mode=train \
    task=$TASK \
    algorithm=grpo \
    algorithm.training.curriculum_schedule=$SCHEDULER \
    model=$MODEL \
    algorithm.training.per_device_train_batch_size=$BATCH_SIZE \
    algorithm.training.scheduler_params.min_prob=0.1 \
    algorithm.training.scheduler_params.temperature=1.0 \
    algorithm.training.scheduler_params.beta=0.7 \
    algorithm.training.scheduler_params.warmup_steps=100 \
    algorithm.training.scheduler_params.vrex_penalty_weight=1.0 \
    algorithm.training.scheduler_params.groupdro_alpha=0.01 \
    algorithm.training.scheduler_params.window_size=100 \
    algorithm.training.max_steps=$MAX_STEPS

echo "Training completed!"