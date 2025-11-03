# Curriculum Reinforcement Learning from Easy to Hard Tasks Improves LLM Reasoning

[![arXiv](https://img.shields.io/badge/arXiv-2506.06632-b31b1b.svg)](https://arxiv.org/abs/2506.06632)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official implementation of the paper **"Curriculum Reinforcement Learning from Easy to Hard Tasks Improves LLM Reasoning"**.

## Overview

This repository implements a curriculum learning framework for training large language models (LLMs) on reasoning tasks using **GRPO (Group Relative Policy Optimization)**. The framework progressively trains models from easy to hard tasks, improving their reasoning capabilities across multiple domains.


## Table of Contents

  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Setup Environment](#setup-environment)
  - [Curriculum Schedules](#curriculum-schedules)
    - [1. Classical](#1-classical)
    - [2. Balanced](#2-balanced)
    - [3. Cosine](#3-cosine)
    - [4. Gaussian](#4-gaussian)
  - [Configuration](#configuration)
    - [Project Structure](#project-structure)
    - [Configuration Structure](#configuration-structure)
  - [Training](#training)
    - [Default Args](#default-args)
    - [Custom Args](#custom-args)
      - [VLLM server setup](#vllm-server-setup)
      - [Training](#training-1)
      - [Evaluation](#evaluation)
  - [Citation](#citation)
  - [License](#license)
  - [Acknowledgments](#acknowledgments)

## Installation

### Prerequisites

- Python 3.10+
- CUDA 12.x compatible GPU
- Conda or Mamba package manager

### Setup Environment

1. Clone the repository:
```bash
git clone https://github.com/divelab/E2H-Reasoning.git
cd E2H-Reasoning
```

2. Create the conda environment:
```bash
bash env/build_env.sh
```

## Curriculum Schedules

The framework supports four curriculum learning schedules:

### 1. Classic
Simple linear progression through tasks based on training progress.

### 2. Balanced
Equal probability for all task difficulty levels throughout training.

### 3. Cosine
Smooth transition from easy to hard tasks using cosine annealing.

### 4. Gaussian
Gaussian distribution with a moving center, transitioning from easy to hard tasks.

**Configuration Example:**
```yaml
algorithm:
  e2h_args:
    curriculum_schedule: gaussian  # Options: classic, balanced, cosine, gaussian
    scheduler_params:
      mu_exp: 0.5
      sigma: 0.5
```

## Configuration
The project uses [Hydra](https://hydra.cc/) for configuration management. Configuration files are located in [config/](config/).

### Project Structure
```
curriculum-reasoning/
├── config/               # Hydra configuration files
│   ├── algorithm/        # Algorithm configs (GRPO)
│   ├── model/            # Model configs (Qwen, Llama)
│   ├── task/             # Task configs (GSM8K, MATH, etc.)
│   └── config.yaml       # Base configuration
├── env/
│   └── environment.yml   # Conda environment specification
├── src/
│   ├── datasets.py       # Dataset loading and preprocessing
│   ├── rewards.py        # Reward function implementations
│   └── trainer.py        # CurriculumGRPOTrainer
├── main.py               # Main entry point for training/testing
├── run.sh                # SLURM submission script
└── README.md             # This file
```

### Configuration Structure
```
config/
├── algorithm/
│   └── grpo.yaml        # GRPO training parameters
├── model/
│   ├── qwen1.5b.yaml    # Qwen 1.5B model config
│   ├── qwen3b.yaml      # Qwen 3B model config
│   └── llama3b.yaml     # Llama 3B model config
├── task/
│   ├── gsm8k.yaml       # GSM8K task config
│   ├── math.yaml        # MATH task config
│   ├── aqua.yaml        # AQUA task config
│   ├── blocksworld.yaml # Blocksworld task config
│   └── countdown.yaml   # Countdown task config
└── config.yaml          # Base configuration
```

## Training

### Default Args
If want to just run our code without modifying any args.
```bash
bash runs.sh \
--model=<qwen1.5b,qwen3b,llama3b> \
--task=<aqua,blocksworld,countdown,gsm8k,math> \
--curriculum_schedule=<classic,balanced,cosine,gaussian>
```

### Custom Args
#### VLLM server setup
If using VLLM server, then execute the following command before training.
```bash
CUDA_VISIBLE_DEVICES=0 \
trl vllm-serve \
--model Qwen/Qwen2.5-1.5B-Instruct \
--dtype bfloat16 \
--max_model_len 4096 \
--trust_remote_code true\
--log_level warning \
&
```
or, VLLM can be run in colocate mode, by changing the configs in `algorithm/grpo.yaml` 

#### Training
```bash
CUDA_VISIBLE_DEVICES=1,2 \
accelerate launch \
--mixed_precision bf16 \
--num_processes 2 \
--dynamo_backend no \
--use_deepspeed \
--zero_stage 3 \
--gradient_accumulation_steps 4 \
--gradient_clipping 1 \
--zero3_init_flag true \
--zero3_save_16bit_model true \
main.py \
mode=train \
model=qwen1.5b \
task=blocksworld \
<ARG Overides>
```

#### Evaluation
```bash
CUDA_VISIBLE_DEVICES=1,2 \
accelerate launch \
--mixed_precision bf16 \
--num_machines 1 \
--num_processes 1 \
--dynamo_backend no \
main.py \
mode=test \
model=qwen1.5b \
task=blocksworld \
<ARG Overides>
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{parashar2025curriculum,
  title={Curriculum Reinforcement Learning from Easy to Hard Tasks Improves LLM Reasoning},
  author={Parashar, Shubham and Gui, Shurui and Li, Xiner and Ling, Hongyi and Vemuri, Sushil and Olson, Blake and Li, Eric and Zhang, Yu and Caverlee, James and Kalathil, Dileep and Ji, Shuiwang},
  journal={arXiv preprint arXiv:2506.06632},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [TRL (Transformer Reinforcement Learning)](https://github.com/huggingface/trl)
- Uses [vLLM](https://github.com/vllm-project/vllm) for efficient inference
- Configuration management via [Hydra](https://hydra.cc/)
- Training optimization with [DeepSpeed](https://github.com/microsoft/DeepSpeed)

---

For questions or issues, please open an issue on GitHub or contact the authors.
