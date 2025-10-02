#!/bin/bash

#SBATCH --job-name=TrainReasoner
#SBATCH --account=ASC25044
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu-a100
#SBATCH --time=0-08:00:00
#SBATCH --overcommit 
#SBATCH --output=logs/%j.log


echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} started ..."


for i in "$@"; do
  case "$i" in
    --model=*)
      model="${i#*=}"
      ;;
    --task=*)
      task="${i#*=}"
      ;;
  esac
done

if [ $model == "qwen1.5b" ]; then
  hf_model="Qwen/Qwen2.5-1.5B-Instruct"
elif [ $model == "qwen3b" ]; then
  hf_model="Qwen/Qwen2.5-3B-Instruct"
fi


source $(conda info --base)/etc/profile.d/conda.sh
conda activate reasoning_env


CUDA_VISIBLE_DEVICES=0 \
trl vllm-serve \
--model $hf_model \
--dtype bfloat16 \
--max_model_len 2048 \
--trust_remote_code true \
&
SERVER_PID=$!


sleep 360


CUDA_VISIBLE_DEVICES=1,2 \
accelerate launch \
--mixed_precision bf16 \
--num_machines 1 \
--num_processes 2 \
--dynamo_backend no \
--use_deepspeed \
--zero_stage 3 \
--gradient_accumulation_steps 4 \
--gradient_clipping 1 \
--zero3_init_flag true \
--zero3_save_16bit_model true \
main.py \
mode=train \
model=$model \
task=$task


kill $SERVER_PID
wait $SERVER_PID 2>/dev/null


CUDA_VISIBLE_DEVICES=1,2 \
accelerate launch \
--num_machines 1 \
--num_processes 1 \
--dynamo_backend no \
main.py \
mode=test \
model=$model \
task=$task


echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} stopped ..."