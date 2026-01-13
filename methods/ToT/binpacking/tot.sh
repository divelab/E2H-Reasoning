# ToT using APIs
python methods/ToT/binpacking/inference.py --base_lm api --n_beam 5 --depth_limit 8 --api_model_id meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo
python methods/ToT/binpacking/inference.py --base_lm api --n_beam 5 --depth_limit 8 --api_model_id meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo
python methods/ToT/binpacking/inference.py --base_lm api --n_beam 5 --depth_limit 8 --api_model_id meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo
python methods/ToT/binpacking/inference.py --base_lm openai --n_beam 5 --depth_limit 8 --openai_model gpt-4o
python methods/ToT/binpacking/inference.py --base_lm openai --n_beam 5 --depth_limit 8 --openai_model gpt-4o-mini


# ToT run locally, uncomment and provide model path
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node 1 --master_port 12345 methods/ToT/binpacking/inference.py --base_lm hf --model_dir /path/to/model --n_beam 10 --depth_limit 5 --calc_reward logits --quanitized None
