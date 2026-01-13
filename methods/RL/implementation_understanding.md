# Implementation Understanding: E2H Curriculum Learning with Variance Regularized Scheduler

This document provides a deep technical understanding of the Easy 2 Hard (E2H) curriculum learning implementation with a focus on the variance regularized scheduler.

## Table of Contents

1. [Core Architecture](#core-architecture)
2. [Task Sampling Mechanism](#task-sampling-mechanism)
3. [Curriculum Schedulers](#curriculum-schedulers)
4. [Variance Regularized Scheduler Deep Dive](#variance-regularized-scheduler-deep-dive)
5. [Reward Feedback Loop](#reward-feedback-loop)
6. [Training Pipeline](#training-pipeline)
7. [Configuration System](#configuration-system)
8. [Data Flow](#data-flow)
9. [Key Implementation Details](#key-implementation-details)

## Core Architecture

The implementation is built around three main components:

1. **TaskSampler**: Controls curriculum learning by managing task sampling probabilities
2. **CurriculumGRPOTrainer**: Extends standard GRPO trainer to integrate curriculum learning
3. **Task-specific Trainers**: Handle different tasks (Countdown, BlocksWorld, Arithmetic, Coding)

### Key Classes

```python
# Core curriculum learning
class TaskSampler(torch.utils.data.Sampler)
class CurriculumGRPOTrainer(GRPOTrainer)

# Task-specific trainers
class CountdownTrainer(BaseTrainer)
class BlocksWorldTrainer(BaseTrainer)
class ArithmeticTrainer(BaseTrainer)
class CodeTrainer(BaseTrainer)
```

## Task Sampling Mechanism

### TaskSampler Overview

The `TaskSampler` is the heart of curriculum learning. It:

1. **Organizes data by difficulty**: Each task difficulty level (0=easiest, N-1=hardest) gets its own data partition
2. **Applies scheduling strategy**: Uses one of 5 scheduling algorithms to compute task probabilities
3. **Samples tasks per batch**: Each batch slot is assigned a task based on computed probabilities
4. **Manages data exhaustion**: Reshuffles task data when exhausted

### Key Implementation Details

```python
# Data organization by task difficulty
task_col = np.array(self.dataset['task'])
self.indices_by_task = {
    t: self.rng.permutation(np.where(task_col == t)[0])
    for t in range(num_tasks)
}

# Batch sampling
for i in range(self.total_iterations):
    probs_dict = self.schedule_func(i, self.total_iterations, self.num_tasks)
    probs = np.array([probs_dict[j] for j in range(self.num_tasks)])
    chosen_tasks = np.random.choice(np.arange(self.num_tasks), size=self.batch_size, p=probs, replace=True)
```

## Curriculum Schedulers

### 1. Balanced Schedule
- **Strategy**: Equal probability for all difficulty levels
- **Formula**: P(task_i) = 1/num_tasks
- **Use case**: Baseline comparison

### 2. Classical Curriculum (Step)
- **Strategy**: Sequential progression through difficulties
- **Formula**: Active task = min(int(t * num_tasks / T), num_tasks - 1)
- **Use case**: Traditional curriculum learning

### 3. Cosine Schedule
- **Strategy**: Smooth transition from easy to hard
- **Formula**: Cosine interpolation between early (easy-focused) and late (hard-focused) distributions
- **Use case**: Gradual curriculum progression

### 4. Gaussian Schedule
- **Strategy**: Bell curve moving from easy to hard
- **Parameters**:
  - `mu_exp`: Controls progression speed
  - `sigma`: Standard deviation of Gaussian
  - `min_prob`: Minimum probability per task
- **Use case**: Smooth curriculum with configurable progression

### 5. Variance Regularized Schedule (VREx)
- **Strategy**: Adaptive based on performance variance
- **Core principle**: Minimize performance variance across tasks for better OOD generalization
- **Use case**: Our main research focus

## Variance Regularized Scheduler Deep Dive

### Theoretical Foundation

Based on two key OOD generalization principles:
1. **VREx (Variance Risk Extrapolation)**: Minimizes performance variance across environments/tasks
2. **GroupDRO**: Focuses on worst-performing groups with exponential weighting

### Implementation Architecture

```python
# Function-level state management
_variance_regularized_schedule.state = {
    'task_performances': {i: deque(maxlen=window_size) for i in range(num_tasks)},
    'task_counts': defaultdict(int),
    'group_weights': np.ones(num_tasks) / num_tasks,
    'current_probs': {i: 1.0 / num_tasks for i in range(num_tasks)},
    'last_update': -1
}
```

### Scoring Mechanism

The scheduler computes sampling scores based on:

1. **Performance Deficit** (30%): `1.0 / (mean_performance + 1e-8)`
2. **Variance Score** (20%): `sqrt(performance_variance + 1e-8)`
3. **Exploration Bonus** (10%): `1.0 / (task_frequency + 1e-8)`
4. **GroupDRO Weight** (30%): Exponentially weighted based on losses
5. **VREx Penalty** (10%): Cross-task variance penalty

### Key Parameters

- `window_size`: Performance tracking window (default: 100)
- `min_prob`: Minimum sampling probability (default: 0.1)
- `temperature`: Softmax temperature (default: 1.0)
- `beta`: Blend factor with uniform distribution (default: 0.5)
- `warmup_steps`: Steps before adaptive sampling (default: 100)
- `vrex_penalty_weight`: Weight for variance penalty (default: 1.0)
- `groupdro_alpha`: Learning rate for group weights (default: 0.01)

## Reward Feedback Loop

### Architecture

The variance regularized scheduler requires performance feedback to make adaptive decisions. This is implemented through a three-stage process:

### Stage 1: Reward Computation (Task-specific)

Each task's reward function computes rewards and stores them in the trainer:

```python
# In each task's reward function (e.g., _countdown_reward_fn)
def _countdown_reward_fn(self, completions, target, numbers, **kwargs):
    rewards = []
    # ... compute rewards for each completion ...
    
    # Store rewards for variance regularized scheduler
    if hasattr(self, 'trainer') and hasattr(self.trainer, 'data_schedule'):
        if self.trainer.data_schedule == 'variance_regularized':
            self.trainer._last_batch_rewards = rewards
    
    return rewards
```

### Stage 2: Task ID Extraction (CurriculumGRPOTrainer)

The trainer extracts task IDs from the current batch:

```python
def training_step(self, model, inputs):
    # Extract task IDs from the batch before processing
    if 'task' in inputs:
        self._current_batch_task_ids = inputs['task'].tolist()
    
    # Call parent training step (this triggers reward computation)
    result = super().training_step(model, inputs)
```

### Stage 3: Performance Update (CurriculumGRPOTrainer)

After the training step, task IDs and rewards are used to update the scheduler:

```python
# Update variance regularized scheduler if using it
if self.data_schedule == 'variance_regularized' and hasattr(self, '_last_batch_rewards'):
    task_ids = self._current_batch_task_ids
    rewards = self._last_batch_rewards
    update_variance_regularized_performance(task_ids, rewards)
```

## Training Pipeline

### Complete Training Flow

1. **Initialization**:
   - Load model and tokenizer
   - Prepare dataset with task annotations
   - Create CurriculumGRPOTrainer with TaskSampler

2. **Each Training Step**:
   - TaskSampler computes task probabilities
   - Batch is sampled according to probabilities
   - Forward pass generates completions
   - Task-specific reward function computes rewards
   - Rewards are stored in trainer
   - GRPO loss is computed and backpropagated
   - Variance regularized scheduler state is updated

3. **Iteration**:
   - Process repeats for max_steps iterations
   - Scheduler adapts probabilities based on performance feedback

### Critical Integration Points

- **Dataset Preparation**: Each sample must have a 'task' field indicating difficulty level
- **Reward Function Integration**: Each task's reward function must store rewards in trainer
- **Task ID Tracking**: CurriculumGRPOTrainer must extract and store task IDs
- **State Management**: Variance regularized scheduler maintains persistent state

## Configuration System

### Hydra Structure

```
conf/
├── config.yaml           # Main configuration
├── algorithm/             # GRPO, PPO configurations
│   ├── grpo.yaml
│   └── grpo_variance_regularized.yaml
├── model/                 # Model configurations
│   ├── qwen15.yaml
│   └── llama.yaml
└── task/                  # Task configurations
    ├── countdown2345.yaml
    └── blocksworld.yaml
```

### Key Configuration Elements

1. **Output Path Generation**: `${model.trim}_${task.name}_${algorithm.name}_${scheduler}_${params}_${max_steps}`
2. **Scheduler Parameters**: Defined in `algorithm.training.scheduler_params`
3. **Task Data Files**: Multiple difficulty levels defined in `task.data_files`

## Data Flow

### Training Data Flow

```
HuggingFace Dataset → Task Annotation → TaskSampler → Batch → Model → Completions → Reward Function → Scheduler Update
```

### Variance Regularized Scheduler Data Flow

```
Task IDs + Rewards → update_variance_regularized_performance() → Performance Tracking → Statistics Computation → Probability Update → Task Sampling
```

## Key Implementation Details

### State Persistence

The variance regularized scheduler uses function attributes for state persistence:

```python
if not hasattr(_variance_regularized_schedule, 'state'):
    _variance_regularized_schedule.state = {...}
```

This ensures state persists across function calls within a training run.

### Warmup Mechanism

During the first `warmup_steps` iterations, the scheduler uses uniform sampling to gather initial performance data before starting adaptive sampling.

### Update Frequency

The scheduler only updates probabilities every 10 steps to avoid too frequent changes that could destabilize training.

### Minimum Probability Floor

All tasks maintain a minimum sampling probability to prevent complete neglect of any difficulty level.

### Data Exhaustion Handling

When a task's data is exhausted, it's automatically reshuffled to ensure continuous training.

### Memory Efficiency

Performance tracking uses `deque` with `maxlen` to automatically manage memory usage and keep only recent performance history.

---

This implementation represents a sophisticated curriculum learning system that adaptively balances exploration and exploitation across difficulty levels while maintaining theoretical grounding in OOD generalization principles.