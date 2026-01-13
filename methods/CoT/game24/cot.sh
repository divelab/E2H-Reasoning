# CoT
python -m torch.distributed.run --master_port 12212 methods/CoT/game24/inference.py --temperature 0.8 --base_lm openai --openai_model gpt-4o-mini
python -m torch.distributed.run --master_port 12212 methods/CoT/game24/inference.py --temperature 0.8 --base_lm openai --openai_model gpt-4o
python -m torch.distributed.run --master_port 12212 methods/CoT/game24/inference.py --temperature 0.8 --base_lm api --api_model meta-llama/Meta-Llama-3.1-8B-Instruct
python -m torch.distributed.run --master_port 12212 methods/CoT/game24/inference.py --temperature 0.8 --base_lm api --api_model meta-llama/Meta-Llama-3.1-70B-Instruct
python -m torch.distributed.run --master_port 12212 methods/CoT/game24/inference.py --temperature 0.8 --base_lm api --api_model meta-llama/Meta-Llama-3.1-405B-Instruct

# Self Consistency, with num 5
python -m torch.distributed.run --master_port 12212 methods/CoT/game24/inference.py --temperature 0.8 --base_lm openai --openai_model gpt-4o-mini --sc_num 5
python -m torch.distributed.run --master_port 12212 methods/CoT/game24/inference.py --temperature 0.8 --base_lm openai --openai_model gpt-4o --sc_num 5
python -m torch.distributed.run --master_port 12212 methods/CoT/game24/inference.py --temperature 0.8 --base_lm api --api_model meta-llama/Meta-Llama-3.1-8B-Instruct --sc_num 5
python -m torch.distributed.run --master_port 12212 methods/CoT/game24/inference.py --temperature 0.8 --base_lm api --api_model meta-llama/Meta-Llama-3.1-70B-Instruct --sc_num 5
python -m torch.distributed.run --master_port 12212 methods/CoT/game24/inference.py --temperature 0.8 --base_lm api --api_model meta-llama/Meta-Llama-3.1-405B-Instruct --sc_num 5
