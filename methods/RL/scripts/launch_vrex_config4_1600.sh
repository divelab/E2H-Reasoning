#!/bin/bash

# VREx Config 4: High Variance Penalty (1600 steps)
# vrex_penalty_weight=2.0, beta=0.7, progression_bias=0.3, performance_threshold=0.6

cd /data/shurui.gui/Projects/gateway/Sys2Bench

WANDB_PROJECT=Sys2Bench \
ROOT_PATH=/data/shurui.gui/Projects/gateway/Sys2Bench \
CUDA_VISIBLE_DEVICES=4,7 \
accelerate launch \
    --num_processes 1 \
    --main_process_port=29764 \
    --config_file methods/RL/deep_speed.yaml \
    methods/RL/main.py \
    mode=train \
    task=countdown2345 \
    algorithm=grpo \
    algorithm.training.curriculum_schedule=variance_regularized \
    model=qwen15 \
    algorithm.training.per_device_train_batch_size=2 \
    algorithm.training.max_steps=1600 \
    algorithm.training.vllm_gpu_memory_utilization=0.6 \
    +algorithm.training.scheduler_params.beta=0.7 \
    +algorithm.training.scheduler_params.vrex_penalty_weight=2.0 \
    +algorithm.training.scheduler_params.progression_bias=0.3 \
    +algorithm.training.scheduler_params.performance_threshold=0.6 \
    > methods/RL/logs/vrex_config4_high_penalty_1600.log 2>&1 &

echo "VREx Config 4 (High Variance Penalty) launched with PID: $!"
echo "Log file: methods/RL/logs/vrex_config4_high_penalty_1600.log"