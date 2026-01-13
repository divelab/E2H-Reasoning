# OpenAI
python -m torch.distributed.run --nproc_per_node 1 --master_port=25678 \
    methods/ToT/AQuA/inference.py \
    --base_lm openai \
    --depth_limit 10  \
    --temperature 0.8 \
    --openai_model gpt-4o-mini \
    --beam_size 10

python -m torch.distributed.run --nproc_per_node 1 --master_port=25678 \
    methods/ToT/AQuA/inference.py \
    --base_lm openai \
    --depth_limit 10  \
    --temperature 0.8 \
    --openai_model gpt-4o \
    --beam_size 10
