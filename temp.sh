#!/bin/bash

#SBATCH --job-name=GenDataset
#SBATCH --account=ASC25087
#SBATCH --nodes=1
#SBATCH --partition=gpu-a100
#SBATCH --time=0-12:00:00
#SBATCH --overcommit 
#SBATCH --output=logs/%j.log


echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} started ..."

source $SCRATCH/miniconda3/etc/profile.d/conda.sh
conda activate reasoning_env

cd $SCRATCH/projects/E2H-Reasoning/datasets

python vllm_split_dataset.py

echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} stopped ..."
