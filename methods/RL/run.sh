
CUDA_VISIBLE_DEVICES=0,4,2,5,7 accelerate launch --num_processes 4 --main_process_port 29600 --config_file methods/RL/deep_speed.yaml  methods/RL/train.py

ROOT_PATH=path_to_Sys2Bench HF_TOKEN=<hf_token_if_not_stored> CUDA_VISIBLE_DEVICES=0,2,3,6,7 accelerate launch --num_processes 4 --config_file methods/RL/deep_speed.yaml  methods/RL/train_blocksworld.py