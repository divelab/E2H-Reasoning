# Day 3
python -m torch.distributed.run --nproc_per_node 1 --master_port=25678 methods/IO/tripplan/inference.py --model_dir openai --temperature 1.0 --base_lm openai --num_cities 3 | tee o1-tripplan-days-3.log

# Day 4
python -m torch.distributed.run --nproc_per_node 1 --master_port=25679 methods/IO/tripplan/inference.py \
  --model_dir openai \
  --temperature 1.0 \
  --base_lm openai \
  --num_cities 4 | tee o1-tripplan-days-4.log

# Day 5
python -m torch.distributed.run --nproc_per_node 1 --master_port=25680 methods/IO/tripplan/inference.py \
  --model_dir openai \
  --temperature 1.0 \
  --base_lm openai \
  --num_cities 5 | tee o1-tripplan-days-5.log

# Day 6
python -m torch.distributed.run --nproc_per_node 1 --master_port=25681 methods/IO/tripplan/inference.py \
  --model_dir openai \
  --temperature 1.0 \
  --base_lm openai \
  --num_cities 6 | tee o1-tripplan-days-6.log

# Day 7
python -m torch.distributed.run --nproc_per_node 1 --master_port=25682 methods/IO/tripplan/inference.py \
  --model_dir openai \
  --temperature 1.0 \
  --base_lm openai \
  --num_cities 7 | tee o1-tripplan-days-7.log

# Day 8
python -m torch.distributed.run --nproc_per_node 1 --master_port=25683 methods/IO/tripplan/inference.py \
  --model_dir openai \
  --temperature 1.0 \
  --base_lm openai \
  --num_cities 8 | tee o1-tripplan-days-8.log

# Day 9
python -m torch.distributed.run --nproc_per_node 1 --master_port=25684 methods/IO/tripplan/inference.py \
  --model_dir openai \
  --temperature 1.0 \
  --base_lm openai \
  --num_cities 9 | tee o1-tripplan-days-9.log

# Day 10
python -m torch.distributed.run --nproc_per_node 1 --master_port=25685 methods/IO/tripplan/inference.py \
  --model_dir openai \
  --temperature 1.0 \
  --base_lm openai \
  --num_cities 10 | tee o1-tripplan-days-10.log

