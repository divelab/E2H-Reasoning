#!/usr/bin/env bash
set -euo pipefail

# Output CSV file
RESULTS="results.csv"
echo "model,task,steps,pass_at_k,accuracy" > "$RESULTS"

# List of models to evaluate
MODELS=(
  "Qwen/Qwen2.5-1.5B-Instruct"
  "Qwen/Qwen2.5-3B-Instruct"
)

# Fixed task name
TASK="blocksworld"

# Iterate over models, steps, and pass_at_k values
for model in "${MODELS[@]}"; do
  # Extract basename (e.g. Qwen2.5-3B-Instruct)
#   model_basename="${model##*/}"
  # Construct the local output directory
#   model_dir="outputs/${model_basename}_blocksworld1246_sgrpo_balanced_0.5_0.5_True_300"

  for steps in 2 4 6 8; do
    for k in 4 16 64 256; do
      echo "Running: model=${model}, steps=${steps}, pass_at_k=${k}"

      # Run the inference script and capture its output
      out=$(CUDA_VISIBLE_DEVICES=5 torchrun --master_port 11211 \
        methods/RL/inference.py \
        --model_dir "$model" \
        --steps "$steps" \
        --pass_at_k "$k")

      # Extract the final Accuracy line (e.g. "Accuracy:  1.0")
      acc=$(printf "%s\n" "$out" | grep -E '^Accuracy:' | tail -n1 | awk '{print $2}')
      echo "Running: model=${model}, steps=${steps}, pass_at_k=${k}, accuracy=${acc}"
      # Append a CSV row: model,task,steps,pass_at_k,accuracy
      echo "${model},${TASK},${steps},${k},${acc}" >> "$RESULTS"
    done
  done
done

echo "All runs complete. Results saved to $RESULTS"
