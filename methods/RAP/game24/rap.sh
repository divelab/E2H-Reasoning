echo 'To RAP run locally, uncomment and provide model path for the commnand.'
# RAP run locally, uncomment and provide model path
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node 1 --master_port 12345 methods/RAP/game24/inference.py --base_lm hf --model_dir /path/to/model --n_beam 5 --depth_limit 4 --calc_reward logits --quanitized None

echo 'RAP ran successfully'