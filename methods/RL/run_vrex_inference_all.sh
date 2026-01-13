#!/bin/bash
# Usage:
#   GPU_IDX=0,1,2,3,4 CONDA_ENV=sys2bench MODEL_TRIM=<model_name> bash ./run_vrex_inference_all.sh
#
# This script launches a new tmux session with one window per countdown difficulty level.
# Each agent will run inference with a specified GPU and test file.
#
# The inference command runs:
#   CUDA_VISIBLE_DEVICES=X ROOT_PATH=/data/shurui.gui/Projects/gateway/Sys2Bench python methods/RL/main.py \
#         mode=inference task=countdown2345 algorithm=grpo model=qwen15 model.family=citrinegui \
#         model.trim=<MODEL_TRIM> task.test_file=<task_file> algorithm.training.max_steps=1600

# Check required environment variables
if [ -z "$GPU_IDX" ]; then
  echo "Error: Please set the GPU_IDX environment variable (comma-separated list). e.g., GPU_IDX=0,1,2,3,4"
  exit 1
fi

if [ -z "$CONDA_ENV" ]; then
  echo "Error: Please set the CONDA_ENV environment variable. e.g., CONDA_ENV=sys2bench"
  exit 1
fi

if [ -z "$MODEL_TRIM" ]; then
  echo "Error: Please set the MODEL_TRIM environment variable. e.g., MODEL_TRIM=Qwen2.5-1.5B-Instruct_countdown2345_grpo_vrex_..."
  exit 1
fi

if [ -z "$MAX_STEPS" ]; then
  MAX_STEPS=1600
fi

# Split the GPU_IDX environment variable into an array
IFS=',' read -r -a gpu_array <<< "$GPU_IDX"

# Test files for different difficulty levels
task_files=(
    "citrinegui/countdown_n2t100_1-100"
    "citrinegui/countdown_n3t100_1-100"
    "citrinegui/countdown_n4t100_1-100"
    "citrinegui/countdown_n5t100_1-100"
    "citrinegui/countdown_n6t100_1-100"
)

# Ensure that the number of GPUs and task files are equal
if [ ${#gpu_array[@]} -ne ${#task_files[@]} ]; then
  echo "Error: The number of GPU indices and task files must be equal."
  echo "Provided ${#gpu_array[@]} GPUs but have ${#task_files[@]} test files."
  exit 1
fi

# Generate a unique session name based on the model and timestamp
session_name="vrex_inference_$(date +%s)"

# Start a new, detached tmux session
tmux new-session -d -s "$session_name"

echo "Launching VREx inference for model: $MODEL_TRIM"
echo "Using GPUs: ${gpu_array[*]}"
echo "Tmux session: $session_name"

# Loop over each test file and create a tmux window for it
for i in "${!gpu_array[@]}"; do
  window_num=$((i + 1))
  difficulty=$((i + 2))  # 2, 3, 4, 5, 6
  
  # For the first inference job, rename the default window; for others, create new windows
  if [ $i -eq 0 ]; then
    tmux rename-window -t "${session_name}:1" "Countdown${difficulty}"
  else
    tmux new-window -t "$session_name" -n "Countdown${difficulty}"
  fi
  
  # Build the inference command
  cmd="VLLM_USE_V1=0 CUDA_VISIBLE_DEVICES=${gpu_array[i]} ROOT_PATH=/data/shurui.gui/Projects/gateway/Sys2Bench python methods/RL/main.py mode=inference task=countdown2345 algorithm=grpo model=qwen15 model.family=citrinegui model.trim=${MODEL_TRIM} task.test_file=${task_files[i]} algorithm.training.max_steps=${MAX_STEPS} task.inference.batch_size=32 2>&1 | tee methods/RL/logs/vrex_inference_countdown${difficulty}.log"
  
  # In each tmux window, activate the conda environment and run the command
  tmux send-keys -t "${session_name}:Countdown${difficulty}" "conda activate ${CONDA_ENV}" C-m
  tmux send-keys -t "${session_name}:Countdown${difficulty}" "${cmd}" C-m
done

echo "All inference jobs launched in tmux session: $session_name"
echo ""
echo "To monitor progress:"
echo "  tmux attach-session -t $session_name"
echo ""
echo "To kill the session:"
echo "  tmux kill-session -t $session_name"