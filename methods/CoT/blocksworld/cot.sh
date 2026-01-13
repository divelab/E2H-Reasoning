# change sc_num for Self Consistency

# OpenAI
for step in {2..12..2}; do
    echo "Running step $step..."
    python -m torch.distributed.run --master_port 18279 methods/CoT/blocksworld/cot_inference.py \
        --base_lm openai \
        --temperature 0.8 \
        --data_path data/blocksworld/split_v1/split_v1_step_${step}_data.json \
        --prompt_path prompts/blocksworld/pool_prompt_v2_step_${step}.json \
        --openai_model gpt-4o-mini
done

for step in {2..12..2}; do
    echo "Running step $step..."
    python -m torch.distributed.run --master_port 18279 methods/CoT/blocksworld/cot_inference.py \
        --base_lm openai \
        --temperature 0.8 \
        --data_path data/blocksworld/split_v1/split_v1_step_${step}_data.json \
        --prompt_path prompts/blocksworld/pool_prompt_v2_step_${step}.json \
        --openai_model gpt-4o
done


# Llama
for step in {2..12..2}; do
    echo "Running step $step..."
    python -m torch.distributed.run --master_port 18279 methods/CoT/blocksworld/cot_inference.py \
        --base_lm api \
        --temperature 0.8 \
        --data_path data/blocksworld/split_v1/split_v1_step_${step}_data.json \
        --prompt_path prompts/blocksworld/pool_prompt_v2_step_${step}.json \
        --api_model_id meta-llama/Meta-Llama-3.1-8B-Instruct
done

for step in {2..12..2}; do
    echo "Running step $step..."
    python -m torch.distributed.run --master_port 18279 methods/CoT/blocksworld/cot_inference.py \
        --base_lm api \
        --temperature 0.8 \
        --data_path data/blocksworld/split_v1/split_v1_step_${step}_data.json \
        --prompt_path prompts/blocksworld/pool_prompt_v2_step_${step}.json \
        --api_model_id meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo
done

for step in {2..12..2}; do
    echo "Running step $step..."
    python -m torch.distributed.run --master_port 18279 methods/CoT/blocksworld/cot_inference.py \
        --base_lm api \
        --temperature 0.8 \
        --data_path data/blocksworld/split_v1/split_v1_step_${step}_data.json \
        --prompt_path prompts/blocksworld/pool_prompt_v2_step_${step}.json \
        --api_model_id meta-llama/Meta-Llama-3.1-405B
done
