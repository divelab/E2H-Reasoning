# Day 1
python -m torch.distributed.run --nproc_per_node 1 --master_port=25678 methods/IO/calendarplan/inference.py \
  --model_dir openai \
  --temperature 0.8 \
  --base_lm openai \
  --num_days 1 | tee o1-calendarplan-days-1.log

# Day 2
python -m torch.distributed.run --nproc_per_node 1 --master_port=25679 methods/IO/calendarplan/inference.py \
  --model_dir openai \
  --temperature 0.8 \
  --base_lm openai \
  --num_days 2 | tee o1-calendarplan-days-2.log

# Day 3
python -m torch.distributed.run --nproc_per_node 1 --master_port=25680 methods/IO/calendarplan/inference.py \
  --model_dir openai \
  --temperature 0.8 \
  --base_lm openai \
  --num_days 3 | tee o1-calendarplan-days-3.log

# Day 4
python -m torch.distributed.run --nproc_per_node 1 --master_port=25681 methods/IO/calendarplan/inference.py \
  --model_dir openai \
  --temperature 0.8 \
  --base_lm openai \
  --num_days 4 | tee o1-calendarplan-days-4.log

# Day 5
python -m torch.distributed.run --nproc_per_node 1 --master_port=25682 methods/IO/calendarplan/inference.py \
  --model_dir openai \
  --temperature 0.8 \
  --base_lm openai \
  --num_days 5 | tee o1-calendarplan-days-5.log
