#!/bin/bash

# OpenAI - change num days 
python -m torch.distributed.run --nproc_per_node 1 --master_port=25678 \
    methods/CoT/calendarplan/inference.py \
    --model_dir openai \
    --temperature 0.8 \
    --base_lm openai \
    --num_days 2 \
    --openai_model gpt-4o-mini

python -m torch.distributed.run --nproc_per_node 1 --master_port=25678 \
    methods/CoT/calendarplan/inference.py \
    --model_dir openai \
    --temperature 0.8 \
    --base_lm openai \
    --num_days 2 \
    --openai_model gpt-4o

# Llama
python -m torch.distributed.run --nproc_per_node 1 --master_port=25678 \
    methods/CoT/calendarplan/inference.py \
    --temperature 0.8 \
    --base_lm api \
    --num_days 2 \
    --api_model_id meta-llama/Meta-Llama-3.1-8B-Instruct

python -m torch.distributed.run --nproc_per_node 1 --master_port=25678 \
    methods/CoT/calendarplan/inference.py \
    --temperature 0.8 \
    --base_lm api \
    --num_days 2 \
    --api_model_id meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo

python -m torch.distributed.run --nproc_per_node 1 --master_port=25678 \
    methods/CoT/calendarplan/inference.py \
    --model_dir meta-llama/Meta-Llama-3.1-405B \
    --temperature 0.8 \
    --base_lm api \
    --num_days 2 \
    --api_model_id meta-llama/Meta-Llama-3.1-405B
