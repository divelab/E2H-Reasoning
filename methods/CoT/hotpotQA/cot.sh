# OpenAI
python -m torch.distributed.run --nproc_per_node 1 --master_port=25678 methods/CoT/hotpotQA/inference.py --model_dir openai --temperature 0.8 --base_lm openai --openai_model gpt-4o-mini

python -m torch.distributed.run --nproc_per_node 1 --master_port=25678 methods/CoT/hotpotQA/inference.py --model_dir openai --temperature 0.8 --base_lm openai --openai_model gpt-4o

# 70B
python -m torch.distributed.run --nproc_per_node 1 --master_port=25678 methods/CoT/hotpotQA/inference.py --temperature 0.8 --base_lm api --api_model_id meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo --n_sc 1

python -m torch.distributed.run --nproc_per_node 1 --master_port=25678 methods/CoT/hotpotQA/inference.py --temperature 0.8 --base_lm api --api_model_id meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo --n_sc 5

# 8B
python -m torch.distributed.run --nproc_per_node 1 --master_port=25678 methods/CoT/hotpotQA/inference.py --temperature 0.8 --base_lm api --api_model_id meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo --n_sc 1

python -m torch.distributed.run --nproc_per_node 1 --master_port=25678 methods/CoT/hotpotQA/inference.py --temperature 0.8 --base_lm api --api_model_id meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo --n_sc 5

# 405 B
python -m torch.distributed.run --nproc_per_node 1 --master_port=25678 methods/CoT/hotpotQA/inference.py --temperature 0.8 --base_lm api --api_model_id meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo --n_sc 1

python -m torch.distributed.run --nproc_per_node 1 --master_port=25678 methods/CoT/hotpotQA/inference.py --temperature 0.8 --base_lm api --api_model_id meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo --n_sc 5
