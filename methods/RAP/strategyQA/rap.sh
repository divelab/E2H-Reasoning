echo 'To RAP run locally, uncomment and provide model path for the commnand.'
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node 1 --master_port 12345 methods/RAP/strategyQA/inference.py --base_lm hf --model_dir /path/to/model --n_action 5 --depth_limit 7 --calc_reward logits --quanitized None --temperature 0.8
echo 'RAP ran successfully'