#!/bin/bash

#pip install "torch==2.5.1" "setuptools<71.0.0"  --index-url https://download.pytorch.org/whl/cu121
#pip install tensorboard
#
#pip install  --upgrade \
#  "transformers==4.48.1" \
#  "datasets==3.1.0" \
#  "accelerate==1.3.0" \
#  "hf-transfer==0.1.9" \
#  "deepspeed==0.15.4" \
#  "trl==0.14.0"

pip install "vllm==0.7.0"
conda install conda-forge::cuda-compiler=12.4.1
pip install nvidia-cuda-nvcc-cu12==12.4.131
pip install trl==0.14.0 deepspeed==0.15.4 hf-transfer==0.1.9 transformers==4.48.1 datasets==3.1.0 accelerate==1.3.0
pip install flash-attn==2.7.3
pip install hydra-core tarski pddl==0.2.0 peft wandb
pip install antlr4-python3-runtime==4.9