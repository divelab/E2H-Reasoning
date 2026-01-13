echo 'To RAP run locally, uncomment and provide model path for the commnand.'
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node 1 --master_port 12345 methods/RAP/prontoqa/inference.py --base_model hf --model_dir /path/to/model --n_action 5 --depth_limit 6 --quantized None
echo 'RAP ran successfully'