echo 'To RAP run locally, uncomment and provide model path for the commnand.'
export CUDA_VISIBLE_DEVICES=0,1
export model_dir="/path/to/model"
# python -m torch.distributed.run --nproc_per_node 1 methods/RAP/blocksworld/inference.py --model_dir $model_dir --steps 2 --log_dir logs/v1_step2 
# python -m torch.distributed.run --nproc_per_node 1 methods/RAP/blocksworld/inference.py --model_dir $model_dir --steps 4 --log_dir logs/v1_step4
# python -m torch.distributed.run --nproc_per_node 1 methods/RAP/blocksworld/inference.py --model_dir $model_dir --steps 6 --log_dir logs/v1_step6
# python -m torch.distributed.run --nproc_per_node 1 methods/RAP/blocksworld/inference.py --model_dir $model_dir --steps 8 --log_dir logs/v1_step8
# python -m torch.distributed.run --nproc_per_node 1 methods/RAP/blocksworld/inference.py --model_dir $model_dir --steps 10 --log_dir logs/v1_step10
# python -m torch.distributed.run --nproc_per_node 1 methods/RAP/blocksworld/inference.py --model_dir $model_dir --steps 12 --log_dir logs/v1_step12
echo 'RAP ran successfully'