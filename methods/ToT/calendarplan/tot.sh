python -m torch.distributed.run --nproc_per_node 1 --master_port=25678 methods/ToT/calendarplan/inference.py --model_dir openai --temperature 0.8 --base_lm openai --num-days 2

CUDA_VISIBLE_DEVICES=4,7,8 python -m torch.distributed.run --nproc_per_node 1 --master_port=12341 methods/ToT/calendarplan/inference.py --base_lm hf --log_dir cube_ToT_LLaMA3p1-8B --calc_reward logits --beam_size 5 --model_dir /data2/eric.li/models/hf/Llama-3.1-8B --quantized None --num-days 2
