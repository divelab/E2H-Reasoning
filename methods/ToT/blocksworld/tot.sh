# Openai
for step in {2..12..2}; do
    echo "Running step $step..."
    python -m torch.distributed.run --master_port 18479 methods/ToT/blocksworld/tot_inference.py \
        --base_lm openai \
        --data_path data/blocksworld/split_v1/split_v1_step_${step}_data.json \
        --depth_limit ${step} \
        --prompt_path prompts/blocksworld/pool_prompt_v2_step_${step}.json --beam_size 10 \
        --temperature 0.8 \
        --openai_model gpt-4o-mini \
        --search_algo beam \
        --reward_aggregator mean
done

for step in {2..12..2}; do
    echo "Running step $step..."
    python -m torch.distributed.run --master_port 18479 methods/ToT/blocksworld/tot_inference.py \
        --base_lm openai \
        --data_path data/blocksworld/split_v1/split_v1_step_${step}_data.json \
        --depth_limit ${step} \
        --prompt_path prompts/blocksworld/pool_prompt_v2_step_${step}.json --beam_size 10 \
        --temperature 0.8 \
        --openai_model gpt-4o \
        --search_algo beam \
        --reward_aggregator mean
done

# Llama

for step in {10..12..2}; do
    echo "Running step $step..."
    python -m torch.distributed.run --master_port 18379 methods/ToT/blocksworld/tot_inference.py \
        --base_lm api \
        --api_model_id meta-llama/Meta-Llama-3.1-8B-Instruct \
        --data_path data/blocksworld/split_v1/split_v1_step_${step}_data.json \
        --depth_limit ${step} \
        --prompt_path prompts/blocksworld/pool_prompt_v2_step_${step}.json --beam_size 10 \
        --temperature 0.8 \
        --search_algo beam \
        --reward_aggregator mean
done

for step in {10..12..2}; do
    echo "Running step $step..."
    python -m torch.distributed.run --master_port 18379 methods/ToT/blocksworld/tot_inference.py \
        --base_lm api \
        --api_model_id meta-llama/Meta-Llama-3.1-70B-Instruct \
        --data_path data/blocksworld/split_v1/split_v1_step_${step}_data.json \
        --depth_limit ${step} \
        --prompt_path prompts/blocksworld/pool_prompt_v2_step_${step}.json --beam_size 10 \
        --temperature 0.8 \
        --search_algo beam \
        --reward_aggregator mean
done

for step in {10..12..2}; do
    echo "Running step $step..."
    python -m torch.distributed.run --master_port 18479 methods/ToT/blocksworld/tot_inference.py \
        --base_lm api \
        --api_model_id meta-llama/Meta-Llama-3.1-405B-Instruct \
        --data_path data/blocksworld/split_v1/split_v1_step_${step}_data.json \
        --depth_limit ${step} \
        --prompt_path prompts/blocksworld/pool_prompt_v2_step_${step}.json --beam_size 10 \
        --temperature 0.8 \
        --search_algo beam \
        --reward_aggregator mean
done
