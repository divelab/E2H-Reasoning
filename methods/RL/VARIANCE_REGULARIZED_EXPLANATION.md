# Variance Regularized Curriculum Scheduler Explained

## Overview

The Variance Regularized Curriculum Scheduler is designed to improve out-of-distribution (OOD) generalization by ensuring the model learns all task difficulties equally well. It combines principles from three key methods:

1. **VREx (Variance Risk Extrapolation)**: Minimizes performance variance across tasks
2. **GroupDRO (Group Distributionally Robust Optimization)**: Focuses on worst-performing groups
3. **Adaptive Curriculum Learning**: Dynamically adjusts task sampling based on performance

## Core Idea

Instead of following a fixed curriculum (easy→hard), this scheduler:
- Monitors performance on each difficulty level
- Identifies which tasks the model struggles with
- Increases sampling probability for underperforming tasks
- Penalizes high variance in performance across tasks

## Key Hyperparameters

### 1. **`min_prob` (default: 0.1)**
- **Purpose**: Ensures every task gets sampled with at least this probability
- **Effect**: Prevents complete neglect of any difficulty level
- **Try**: 0.05 (more focused) or 0.2 (more uniform)

### 2. **`temperature` (default: 1.0)**
- **Purpose**: Controls how "sharp" the probability distribution is
- **Effect**: Lower = more deterministic, Higher = more uniform
- **Try**: 0.5 (sharper focus) or 2.0 (smoother distribution)

### 3. **`beta` (default: 0.5)**
- **Purpose**: Blending factor between uniform and adaptive sampling
- **Effect**: 0 = fully uniform, 1 = fully adaptive
- **Try**: 0.3 (more uniform) or 0.7 (more adaptive)

### 4. **`warmup_steps` (default: 100)**
- **Purpose**: Steps before adaptive sampling kicks in
- **Effect**: Allows initial uniform exploration
- **Try**: 50 (faster adaptation) or 200 (more initial exploration)

### 5. **`vrex_penalty_weight` (default: 1.0)**
- **Purpose**: Weight for variance minimization objective
- **Effect**: Higher = stronger push for uniform performance
- **Try**: 0.5 (less emphasis) or 2.0 (strong emphasis)

### 6. **`groupdro_alpha` (default: 0.01)**
- **Purpose**: Learning rate for worst-group weighting
- **Effect**: Higher = faster adaptation to worst groups
- **Try**: 0.005 (slower) or 0.02 (faster)

### 7. **`window_size` (default: 100)**
- **Purpose**: Number of recent samples to consider for statistics
- **Effect**: Smaller = more reactive, Larger = more stable
- **Try**: 50 (more reactive) or 200 (more stable)

## How It Works

### Step 1: Performance Tracking
```python
# For each task, we track:
- Recent performance scores (rewards)
- Number of times sampled
- Running statistics (mean, variance)
```

### Step 2: Score Calculation
Each task gets a score based on five factors:

```python
score = 0.3 * performance_deficit +    # Focus on poorly performing tasks
        0.2 * variance_score +          # Tasks with unstable performance
        0.1 * exploration_bonus +       # Under-sampled tasks
        0.3 * groupdro_weight +         # Worst-group emphasis
        0.1 * vrex_penalty              # Cross-task variance penalty
```

### Step 3: Probability Assignment
```python
1. Apply softmax with temperature to scores
2. Blend with uniform distribution using beta
3. Enforce minimum probability constraints
4. Normalize to sum to 1
```

## Testing Different Regularization Methods

### 1. **Pure VREx Mode**
Focus only on minimizing variance across tasks:
```python
algorithm.training.scheduler_params:
  vrex_penalty_weight: 2.0   # High emphasis on variance
  groupdro_alpha: 0.0        # Disable GroupDRO
  beta: 0.8                  # High adaptivity
  temperature: 0.5           # Sharp focus
```

### 2. **Pure GroupDRO Mode**
Focus only on worst-performing groups:
```python
algorithm.training.scheduler_params:
  vrex_penalty_weight: 0.0   # Disable VREx
  groupdro_alpha: 0.02       # Strong GroupDRO
  beta: 0.9                  # Very adaptive
  min_prob: 0.05             # Allow strong focus
```

### 3. **Balanced Mode** (Default)
Combine both approaches:
```python
algorithm.training.scheduler_params:
  vrex_penalty_weight: 1.0   # Standard VREx
  groupdro_alpha: 0.01       # Standard GroupDRO
  beta: 0.5                  # Balanced blending
  temperature: 1.0           # Moderate sharpness
```

### 4. **Conservative Mode**
More uniform sampling with mild adaptation:
```python
algorithm.training.scheduler_params:
  beta: 0.3                  # More uniform
  min_prob: 0.15             # Higher minimum
  temperature: 2.0           # Smoother distribution
  warmup_steps: 200          # Longer warmup
```

### 5. **Aggressive Mode**
Strong adaptation to performance:
```python
algorithm.training.scheduler_params:
  beta: 0.9                  # Highly adaptive
  min_prob: 0.02             # Allow strong focus
  temperature: 0.3           # Very sharp
  groupdro_alpha: 0.03       # Fast adaptation
  window_size: 50            # Quick reactions
```

## Running Experiments

### Basic Command Structure
```bash
WANDB_PROJECT=Sys2Bench ROOT_PATH=/data/shurui.gui/Projects/Sys2Bench \
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --config_file methods/RL/deep_speed.yaml \
    methods/RL/main.py \
    mode=train \
    task=countdown6 \
    algorithm=grpo \
    algorithm.training.curriculum_schedule=variance_regularized \
    model=qwen15 \
    algorithm.training.max_steps=1600 \
    algorithm.training.scheduler_params.KEY=VALUE
```

### Example: Testing Pure VREx
```bash
# High variance penalty, no GroupDRO
algorithm.training.scheduler_params.vrex_penalty_weight=2.0 \
algorithm.training.scheduler_params.groupdro_alpha=0.0 \
algorithm.training.scheduler_params.beta=0.8
```

### Example: Testing Pure GroupDRO
```bash
# No variance penalty, strong GroupDRO
algorithm.training.scheduler_params.vrex_penalty_weight=0.0 \
algorithm.training.scheduler_params.groupdro_alpha=0.02 \
algorithm.training.scheduler_params.beta=0.9
```

## Monitoring Performance

Watch for these metrics in WandB:
1. **Per-task accuracy**: Should become more uniform over time
2. **Cross-task variance**: Should decrease
3. **Worst-group performance**: Should improve
4. **Sampling probabilities**: Should adapt based on performance

## Tips for Experimentation

1. **Start with default parameters** to establish baseline
2. **Change one parameter at a time** to understand effects
3. **Use shorter runs** (800 steps) for initial tests
4. **Monitor early behavior** - if adaptation is too slow/fast, adjust accordingly
5. **Save checkpoints** at different stages to analyze learning progression

## Expected Behavior

- **Early training**: Uniform sampling during warmup
- **Mid training**: Adaptation kicks in, focusing on struggling tasks
- **Late training**: Should achieve more balanced performance across all difficulties
- **Final result**: Lower variance in task performance = better OOD generalization

## Debugging

If the scheduler isn't working as expected:
1. Check if performances are being updated properly
2. Verify task IDs match expected range
3. Monitor sampling probabilities over time
4. Look for numerical instabilities (NaN/Inf)

Remember: The goal is to achieve similar performance across all difficulty levels, not just high average performance!