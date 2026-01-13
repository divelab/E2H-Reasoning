#!/bin/bash

#SBATCH --job-name=TrainReasoner
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu-a100
#SBATCH --time=0-24:00:00
#SBATCH --overcommit 
#SBATCH --output=logs/%j.log



echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} started ..."


mu="0.5"
sigma="0.5"
for i in "$@"; do
  case "$i" in
    --task=*)
      task="${i#*=}"
      ;;
    --schedule=*)
      schedule="${i#*=}"
      ;;
    --mu=*)
      mu="${i#*=}"
      ;;
    --sigma=*)
      sigma="${i#*=}"
      ;;
  esac
done


if [ $task == "math" ]; then
  max_steps="1200"
elif [ $task == "aqua" ]; then
  max_steps="1600"
elif [ $task == "gsm8k" ]; then
  max_steps="1600"
else
  max_steps=1600
fi


source $SCRATCH/miniconda3/etc/profile.d/conda.sh
conda activate reasoning_env
cd $SCRATCH/projects/Sys2Bench_shurui


CUDA_VISIBLE_DEVICES=0 \
trl vllm-serve \
--model Qwen/Qwen2.5-1.5B-Instruct \
--dtype bfloat16 \
--max_model_len 4096 \
--trust_remote_code True \
&
SERVER_PID=$!


sleep 360


CUDA_VISIBLE_DEVICES=1,2 \
ROOT_PATH="$SCRATCH/projects/Sys2Bench_shurui" \
accelerate launch \
--mixed_precision bf16 \
--num_machines 1 \
--num_processes 2 \
--dynamo_backend no \
--use_deepspeed \
--zero_stage 3 \
--gradient_accumulation_steps 4 \
--zero3_init_flag true \
--zero3_save_16bit_model true \
methods/RL/main.py \
mode=train \
model=qwen15 \
task=$task \
algorithm=grpo \
algorithm.training.curriculum_schedule=$schedule \
algorithm.training.scheduler_params.mu_exp=$mu \
algorithm.training.scheduler_params.sigma=$sigma \
algorithm.training.max_steps=$max_steps \
algorithm.training.curriculum=false \
algorithm.training.vllm_gpu_memory_utilization=0.8 \
algorithm.training.vllm_mode=server \
algorithm.training.report_to=\[tensorboard\] \
algorithm.training.push_to_hub=false \
algorithm.training.save_strategy=no


kill $SERVER_PID
wait $SERVER_PID 2>/dev/null


CUDA_VISIBLE_DEVICES=1,2 \
ROOT_PATH="$SCRATCH/projects/Sys2Bench_shurui" \
accelerate launch \
--mixed_precision bf16 \
--num_machines 1 \
--num_processes 1 \
--dynamo_backend no \
methods/RL/main.py \
mode=inference \
model=qwen15 \
task=$task \
algorithm=grpo \
algorithm.training.curriculum_schedule=$schedule \
algorithm.training.scheduler_params.mu_exp=$mu \
algorithm.training.scheduler_params.sigma=$sigma \
algorithm.training.max_steps=$max_steps \
algorithm.training.curriculum=false \
algorithm.training.vllm_gpu_memory_utilization=0.8 \
algorithm.training.vllm_mode=server \
algorithm.training.report_to=\[tensorboard\] \
algorithm.training.push_to_hub=false \
algorithm.training.save_strategy=no


echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} stopped ..."
