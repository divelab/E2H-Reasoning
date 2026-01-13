
python -m torch.distributed.run --nproc_per_node 1 --master_port=25678 methods/ToT/cube/inference.py --model_dir openai --temperature 1.0 --base_lm openai --openai_model gpt-4o-mini

CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.run --nproc_per_node 1 --master_port=12345 methods/ToT/cube/inference.py --base_lm hf --log_dir cube_ToT_LLaMA3p1-8B --calc_reward logits --beam_size 5 --model_dir /nvme-data/shparashar/models/hf/Llama-3.1-8B --quantized None

CUDA_VISIBLE_DEVICES=3,5,8 python -m torch.distributed.run --nproc_per_node 1 --master_port=12345 methods/ToT/cube/inference.py --base_lm hf --log_dir cube_ToT_LLaMA3p1-8B --beam_size 5 --model_dir /data2/eric.li/models/hf/Llama-3.1-8B --quantized None
