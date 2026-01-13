#!/bin/bash
# Usage:
#   GPU_IDX=0,1 bash ./tmux_launch.sh <conda_env>
#
# This script launches a new tmux session with one window per agent.
# Each agent will run the testing command with a specified GPU, a hard-coded model.trim,
# and a hard-coded task.test_file.
#
# The testing command is:
#   CUDA_VISIBLE_DEVICES=X ROOT_PATH=/data/shurui.gui/Projects/Sys2Bench python methods/RL/main.py \
#         mode=inference task=countdown345 algorithm=grpo model=qwen15 model.family=citrinegui \
#         model.trim=<model_trim> task.test_file=<task_file>
#
# Edit the model_trims and task_files arrays to suit your needs.

# Make sure the GPU_IDX environment variable is provided.
if [ -z "$GPU_IDX" ]; then
  echo "Error: Please set the GPU_IDX environment variable (comma-separated list). e.g., GPU_IDX=0,1"
  exit 1
fi

# Split the GPU_IDX environment variable into an array.
IFS=',' read -r -a gpu_array <<< "$GPU_IDX"

# Hard-coded list of model trims (adjust as needed)
model_trims=(
  "Qwen2.5-1.5B-Instruct_countdown2345_grpo_gaussian_0.5_0.5_True_1600"
  "Qwen2.5-1.5B-Instruct_countdown2345_grpo_gaussian_0.5_0.5_True_1600"
  "Qwen2.5-1.5B-Instruct_countdown2345_grpo_gaussian_0.5_0.5_True_1600"
  "Qwen2.5-1.5B-Instruct_countdown2345_grpo_gaussian_0.5_0.5_True_1600"
  "Qwen2.5-1.5B-Instruct_countdown2345_grpo_gaussian_0.5_0.5_True_1600"
)

# Hard-coded list of task files (adjust as needed)
task_files=(
  "citrinegui/countdown_n2t100_1-100"
  "citrinegui/countdown_n3t100_1-100"
  "citrinegui/countdown_n4t100_1-100"
  "citrinegui/countdown_n5t100_1-100"
  "citrinegui/countdown_n6t100_1-100"
)

# Ensure that the number of GPUs, model trims, and task files are equal.
if [ ${#gpu_array[@]} -ne ${#model_trims[@]} ] || [ ${#gpu_array[@]} -ne ${#task_files[@]} ]; then
  echo "Error: The number of GPU indices, model trims, and task files must all be equal."
  exit 1
fi

# Check that the conda environment name is provided as the first argument.
if [ -z "$1" ]; then
  echo "Error: Please provide the name of the conda environment as the first argument."
  exit 1
fi

conda_env=$1
evaluate_step=$2

# Generate a unique session name based on the conda environment name and a timestamp.
session_name="exps_${conda_env}_$(date +%s)"

# Start a new, detached tmux session.
tmux new-session -d -s "$session_name"

# Loop over each parameter set and create a tmux window for it.
for i in "${!gpu_array[@]}"; do
  window_num=$((i + 1))

  # For the first agent, rename the default window; for others, create new windows.
  if [ $i -eq 0 ]; then
    tmux rename-window -t "${session_name}:1" "Agent ${window_num}"
  else
    tmux new-window -t "$session_name" -n "Agent ${window_num}"
  fi

  # Build the command using the corresponding GPU, model trim, and task file.
  cmd="CUDA_VISIBLE_DEVICES=${gpu_array[i]} ROOT_PATH=/data/shurui.gui/Projects/Sys2Bench python methods/RL/main.py mode=inference task=countdown2345 algorithm=grpo model=qwen15 model.family=citrinegui model.trim=${model_trims[i]} task.test_file=${task_files[i]} algorithm.training.max_steps=${evaluate_step} task.inference.temperature=0.7 task.inference.pass_at_k=256"

  # In each tmux window, activate the conda environment and run the command.
  tmux send-keys -t "${session_name}:Agent ${window_num}" "conda activate ${conda_env}" C-m
  tmux send-keys -t "${session_name}:Agent ${window_num}" "${cmd}" C-m
done

echo "Launched tmux session: $session_name"