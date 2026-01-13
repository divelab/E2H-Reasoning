# OpenAI
for step in {3..10..1}; do
    echo "Running step $step..."
    python -m torch.distributed.run --master_port 12079 methods/CoT/tripplan/inference.py \
        --base_lm openai \
        --temperature 0.8 \
        --num_cities $step \
        --openai_model gpt-4o-mini
done

for step in {3..10..1}; do
    echo "Running step $step..."
    python -m torch.distributed.run --master_port 12079 methods/CoT/tripplan/inference.py \
        --base_lm openai \
        --temperature 0.8 \
        --num_cities $step \
        --openai_model gpt-4o
done

# Llama
for step in {3..10..1}; do
    echo "Running step $step..."
    python -m torch.distributed.run --master_port 12079 methods/CoT/tripplan/inference.py \
        --base_lm api \
        --temperature 0.8 \
        --num_cities $step \
        --api_model_id meta-llama/Meta-Llama-3.1-8B-Instruct
done

for step in {3..10..1}; do
    echo "Running step $step..."
    python -m torch.distributed.run --master_port 12079 methods/CoT/tripplan/inference.py \
        --base_lm api \
        --temperature 0.8 \
        --num_cities $step \
        --api_model_id meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo
done

for step in {3..10..1}; do
    echo "Running step $step..."
    python -m torch.distributed.run --master_port 12079 methods/CoT/tripplan/inference.py \
        --base_lm api \
        --temperature 0.8 \
        --num_cities $step \
        --api_model_id meta-llama/Meta-Llama-3.1-405B
done
