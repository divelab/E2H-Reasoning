# OpenAI
for city in {3..10}; do
    echo "Running step $city..."
    python -m torch.distributed.run --master_port 11279 methods/ToT/tripplan/inference.py \
        --base_lm openai \
        --num_cities ${city} \
        --depth_limit $((city * 2)) \
        --beam_size 10 \
        --openai_model gpt-4o-mini \
        --temperature 0.8
done 

for city in {3..10}; do
    echo "Running step $city..."
    python -m torch.distributed.run --master_port 11279 methods/ToT/tripplan/inference.py \
        --base_lm openai \
        --num_cities ${city} \
        --depth_limit $((city * 2)) \
        --beam_size 10 \
        --openai_model gpt-4o \
        --temperature 0.8
done 

# Llama
for city in {3..10}; do
    echo "Running step $city..."
    python -m torch.distributed.run --master_port 11279 methods/ToT/tripplan/inference.py \
        --base_lm api \
        --num_cities ${city} \
        --depth_limit $((city * 2)) \
        --beam_size 10 \
        --api_model_id meta-llama/Meta-Llama-3.1-8B-Instruct \
        --temperature 0.8
done 

for city in {3..10}; do
    echo "Running step $city..."
    python -m torch.distributed.run --master_port 11279 methods/ToT/tripplan/inference.py \
        --base_lm api \
        --num_cities ${city} \
        --depth_limit $((city * 2)) \
        --beam_size 10 \
        --api_model_id meta-llama/Meta-Llama-3.1-70B-Instruct \
        --temperature 0.8
done 

for city in {3..10}; do
    echo "Running step $city..."
    python -m torch.distributed.run --master_port 11279 methods/ToT/tripplan/inference.py \
        --base_lm api \
        --num_cities ${city} \
        --depth_limit $((city * 2)) \
        --beam_size 10 \
        --api_model_id meta-llama/Meta-Llama-3.1-405B-Instruct \
        --temperature 0.8
done 
