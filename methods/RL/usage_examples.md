# Usage Examples

This document provides detailed examples for using the RL training framework with various configurations.

## Basic Usage

The framework uses Hydra for configuration, allowing for easy command-line overrides.

### Training Examples

#### Blocksworld with GRPO

Train a Qwen model on the blocksworld task using GRPO:

```bash
python hydra_train_inference.py mode=train task=blocksworld algorithm=grpo
```

#### Countdown with PPO

Train a Qwen model on the countdown task using PPO:

```bash
python hydra_train_inference.py mode=train task=countdown algorithm=ppo
```

### Inference Examples

#### Blocksworld Inference

Run inference with a trained model on blocksworld:

```bash
python hydra_train_inference.py mode=inference task=blocksworld algorithm=grpo
```

#### Countdown Inference

Run inference with a trained model on countdown:

```bash
python hydra_train_inference.py mode=inference task=countdown algorithm=ppo
```

## Advanced Configuration

Hydra allows overriding any configuration value directly from the command line:

### Model Configuration

Use a different model or configuration:

```bash
python hydra_train_inference.py model.name="Qwen/Qwen2.5-7B-Instruct" model.torch_dtype="float16"
```

### Training Parameters

Adjust training parameters:

```bash
python hydra_train_inference.py algorithm.training.learning_rate=2e-6 algorithm.training.max_steps=500
```

### Dataset Size

Control the dataset size:

```bash
python hydra_train_inference.py experiment.dataset_size=1000 experiment.test_size=0.2
```

### LoRA Configuration

Customize LoRA parameters:

```bash
python hydra_train_inference.py lora.r=64 lora.alpha=128
```

## Combining Multiple Tasks and Algorithms

You can create specialized configurations combining different tasks and algorithms:

### Creating a Custom Run Configuration

1. Create a custom configuration file (e.g., `conf/custom_runs/blocksworld_ppo.yaml`):

```yaml
# Example custom configuration
defaults:
  - /task: blocksworld
  - /algorithm: ppo

experiment:
  dataset_size: 500

algorithm:
  training:
    learning_rate: 2e-5
    max_steps: 400
```

2. Run with the custom configuration:

```bash
python hydra_train_inference.py --config-name custom_runs/blocksworld_ppo
```

## Multi-Run Experiments

Run multiple training configurations in parallel using Hydra's multirun functionality:

```bash
python hydra_train_inference.py -m task=blocksworld,countdown algorithm=grpo,ppo
```

This will run all four combinations of tasks and algorithms.

## Debugging Training

For debugging purposes, you can run with a very small dataset:

```bash
python hydra_train_inference.py experiment.dataset_size=10 algorithm.training.max_steps=5
```

## Using GPU Management

If you want to keep the GPU memory allocated after training:

```bash
python hydra_train_inference.py occupy_gpu_memory=true occupy_gpu_memory_gb=40
```

## Output Directory Structure

After running training or inference, you'll find outputs organized as follows:

```
outputs/
├── Qwen2.5-3B-Instruct_blocksworld_grpo_20250319120000/
│   ├── config_dump.yaml
│   ├── blocksworld_grpo/  # Model checkpoints
│   │   ├── checkpoint-50
│   │   ├── checkpoint-100
│   │   ├── ...
│   │   └── pytorch_model.bin
│   └── inference_results.json  # If inference was run
│
├── Qwen2.5-3B-Instruct_countdown_ppo_20250319130000/
│   └── ...
...
```