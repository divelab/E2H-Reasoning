# OpenAI
python -m torch.distributed.run --nproc_per_node 1 --master_port=25678 \
    methods/ToT/prontoqa/tot_inference.py \
    --base_lm openai \
    --depth_limit 10  \
    --temperature 0.8 \
    --openai_model gpt-4o-mini \
    --beam_size 10

python -m torch.distributed.run --nproc_per_node 1 --master_port=25678 \
    methods/ToT/prontoqa/tot_inference.py \
    --base_lm openai \
    --depth_limit 10  \
    --temperature 0.8 \
    --openai_model gpt-4o \
    --beam_size 10

# Llama
python -m torch.distributed.run --nproc_per_node 1 --master_port=25678 \
    methods/ToT/prontoqa/tot_inference.py \
    --base_lm api \
    --depth_limit 10  \
    --temperature 0.8 \
    --beam_size 10 \
    --api_model_id meta-llama/Meta-Llama-3.1-8B-Instruct

python -m torch.distributed.run --nproc_per_node 1 --master_port=25678 \
    methods/ToT/prontoqa/tot_inference.py \
    --base_lm api \
    --depth_limit 10  \
    --temperature 0.8 \
    --beam_size 10 \
    --api_model_id meta-llama/Meta-Llama-3.1-70B-Instruct

python -m torch.distributed.run --nproc_per_node 1 --master_port=25678 \
    methods/ToT/prontoqa/tot_inference.py \
    --base_lm api \
    --depth_limit 10  \
    --temperature 0.8 \
    --beam_size 10 \
    --api_model_id meta-llama/Meta-Llama-3.1-405B-Instruct
