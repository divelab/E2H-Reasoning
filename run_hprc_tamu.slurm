#!/bin/bash

#SBATCH --job-name=TrainReasoner
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:3
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-12:00:00
#SBATCH --overcommit 
#SBATCH --output=logs/%j.log


echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} started ..."


mu="0.5"
sigma="0.5"
gradaccumsteps=4
for i in "$@"; do
  case "$i" in
    --model=*)
      model="${i#*=}"
      ;;
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
    --gradaccumsteps=*)
      gradaccumsteps="${i#*=}"
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


ml purge
ml WebProxy
ml GCCcore/13.3.0
ml Miniconda3/23.10.0-1
ml CUDA/12.6.0
ml NCCL/2.22.3-CUDA-12.6.0
ml binutils/2.42
source /sw/eb/sw/Miniconda3/23.10.0-1/etc/profile.d/conda.sh
conda activate sys2bench_env
cd $SCRATCH/projects/Sys2Bench
MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))


ROOT_PATH="$SCRATCH/projects/Sys2Bench" \
accelerate launch \
--mixed_precision bf16 \
--num_processes 2 \
--num_machines 1 \
--gpu_ids 0,1,2 \
--dynamo_backend no \
--use_deepspeed \
--main_process_port $MASTER_PORT \
--zero_stage 3 \
--offload_optimizer_device none \
--offload_param_device none \
--gradient_accumulation_steps $gradaccumsteps \
--zero3_init_flag true \
--zero3_save_16bit_model true \
methods/RL/main.py \
mode=train \
model="$model" \
task="$task" \
algorithm.training.curriculum_schedule="$schedule" \
algorithm.training.scheduler_params.mu_exp="$mu" \
algorithm.training.scheduler_params.sigma="$sigma" \
algorithm.training.max_steps="$max_steps" \
algorithm.training.per_device_train_batch_size="$((8 / gradaccumsteps))" \
algorithm.training.gradient_accumulation_steps=$gradaccumsteps \
algorithm.training.curriculum=false \
algorithm.training.vllm_gpu_memory_utilization=0.8 \
algorithm.training.report_to=\[tensorboard\] \
algorithm.training.push_to_hub=false \
algorithm.training.save_strategy=no


ROOT_PATH="$SCRATCH/projects/Sys2Bench" \
accelerate launch \
--mixed_precision bf16 \
--num_processes 1 \
--num_machines 1 \
--gpu_ids 0,1 \
--main_process_port $MASTER_PORT \
--dynamo_backend no \
methods/RL/main.py \
mode=inference \
model="$model" \
task="$task" \
algorithm.training.curriculum_schedule="$schedule" \
algorithm.training.scheduler_params.mu_exp="$mu" \
algorithm.training.scheduler_params.sigma="$sigma" \
algorithm.training.max_steps="$max_steps" \
algorithm.training.per_device_train_batch_size="$((8 / gradaccumsteps))" \
algorithm.training.gradient_accumulation_steps=$gradaccumsteps \
algorithm.training.curriculum=false \
algorithm.training.vllm_gpu_memory_utilization=0.8 \
algorithm.training.report_to=\[tensorboard\] \
algorithm.training.push_to_hub=false \
algorithm.training.save_strategy=no


echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} stopped ..."
