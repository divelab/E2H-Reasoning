#!/bin/bash

# VREx Config 3: Conservative Progression (1600 steps)
# beta=0.6, progression_bias=0.2, performance_threshold=0.7

cd /data/shurui.gui/Projects/gateway/Sys2Bench

WANDB_PROJECT=Sys2Bench \
ROOT_PATH=/data/shurui.gui/Projects/gateway/Sys2Bench \
CUDA_VISIBLE_DEVICES=3,4 \
accelerate launch \
    --num_processes 1 \
    --main_process_port=29763 \
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
    +algorithm.training.scheduler_params.beta=0.6 \
    +algorithm.training.scheduler_params.progression_bias=0.2 \
    +algorithm.training.scheduler_params.performance_threshold=0.7 \
    > methods/RL/logs/vrex_config3_conservative_1600.log 2>&1 &

echo "VREx Config 3 (Conservative) launched with PID: $!"
echo "Log file: methods/RL/logs/vrex_config3_conservative_1600.log"

