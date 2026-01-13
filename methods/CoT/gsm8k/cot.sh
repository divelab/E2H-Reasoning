# change sc_num for Self Consistency

# OpenAI
python -m torch.distributed.run --nproc_per_node 1 --master_port=25678 \
    methods/CoT/gsm8k/inference.py \
    --base_lm openai \
    --temperature 0.8 \
    --openai_model gpt-4o-mini

python -m torch.distributed.run --nproc_per_node 1 --master_port=25678 \
    methods/CoT/gsm8k/inference.py \
    --base_lm openai \
    --temperature 0.8 \
    --openai_model gpt-4o


# Llama
python -m torch.distributed.run --nproc_per_node 1 --master_port=25678 \
    methods/CoT/gsm8k/inference.py \
    --base_lm api \
    --temperature 0.8 \
    --api_model_id meta-llama/Meta-Llama-3.1-8B-Instruct


python -m torch.distributed.run --nproc_per_node 1 --master_port=25678 \
    methods/CoT/gsm8k/inference.py \
    --base_lm api \
    --temperature 0.8 \
    --api_model_id meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo

python -m torch.distributed.run --nproc_per_node 1 --master_port=25678 \
    methods/CoT/gsm8k/inference.py \
    --base_lm api \
    --temperature 0.8 \
    --api_model_id meta-llama/Meta-Llama-3.1-405B
