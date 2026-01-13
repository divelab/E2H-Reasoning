# Variance Regularized Scheduler - Integration Guide

## Current Status

The variance regularized scheduler is implemented but needs proper integration with the reward collection system. Here's what needs to be done:

## Missing Piece: Reward Collection

Currently, the `_last_batch_rewards` attribute is referenced but never set. This needs to be implemented in the `CurriculumGRPOTrainer.training_step()` method.

## Required Implementation

### Option 1: Override the training_step method more completely

```python
def training_step(self, model, inputs):
    # Call parent training step
    outputs = super().training_step(model, inputs)
    
    # Extract rewards from the outputs
    if "rewards" in outputs:
        # Store rewards for variance regularized scheduler
        self._last_batch_rewards = outputs["rewards"].detach().cpu().numpy()
    elif hasattr(self, "current_batch_rewards"):
        # Alternative: if rewards are stored elsewhere
        self._last_batch_rewards = self.current_batch_rewards
    
    # Update variance regularized scheduler if using it
    if self.data_schedule == 'variance_regularized' and hasattr(self, '_last_batch_rewards'):
        sampler = self._get_train_sampler()
        if hasattr(sampler, 'last_chosen_tasks'):
            task_ids = sampler.last_chosen_tasks.tolist()
            rewards = self._last_batch_rewards
            update_variance_regularized_performance(task_ids, rewards)
    
    return outputs
```

### Option 2: Hook into the compute_rewards method

```python
def compute_rewards(self, *args, **kwargs):
    # Call parent compute_rewards
    rewards = super().compute_rewards(*args, **kwargs)
    
    # Store for variance regularized scheduler
    self._last_batch_rewards = rewards.detach().cpu().numpy()
    
    return rewards
```

### Option 3: Use the log_stats method

The TRL library's trainers typically log rewards in their `log_stats` method. We could override this:

```python
def log_stats(self, stats, batch, rewards, columns_to_log):
    # Store rewards for variance regularized scheduler
    if rewards is not None:
        self._last_batch_rewards = rewards.detach().cpu().numpy()
    
    # Call parent log_stats
    return super().log_stats(stats, batch, rewards, columns_to_log)
```

## Recommended Approach

Based on TRL's GRPO implementation, the rewards are typically available in the training loop. The cleanest approach would be:

1. **Check TRL's GRPOTrainer source** to see exactly where rewards are computed
2. **Override the appropriate method** (likely `step` or `training_step`)
3. **Extract rewards** from the appropriate data structure
4. **Call update function** with task IDs and rewards

## Temporary Workaround

If you want to test the scheduler without full integration, you can simulate reward updates:

```python
# In training_step
if self.data_schedule == 'variance_regularized':
    # Simulate rewards for testing
    batch_size = inputs['input_ids'].shape[0]
    fake_rewards = np.random.rand(batch_size)  # Replace with actual rewards
    
    sampler = self._get_train_sampler()
    if hasattr(sampler, 'last_chosen_tasks'):
        task_ids = sampler.last_chosen_tasks.tolist()
        update_variance_regularized_performance(task_ids, fake_rewards)
```

## Finding the Actual Rewards

In GRPO training, rewards are typically:
1. Computed by a reward model after generation
2. Stored in the batch or stats dictionary
3. Used to compute advantages for policy gradient

Look for these in the parent class:
- `self.reward_model`
- `batch['rewards']`
- `stats['rewards']`
- Return values from `generate` or `compute_rewards`

## Next Steps

1. **Inspect TRL's GRPOTrainer** source code to understand the exact flow
2. **Add logging** to see what data is available in training_step
3. **Implement proper reward extraction** based on the actual data structure
4. **Test with real training** to ensure rewards are being tracked correctly

## Verification

To verify the integration is working:

```python
# Add this to training_step
if self.state.global_step % 100 == 0 and self.data_schedule == 'variance_regularized':
    state = _variance_regularized_schedule.state
    print(f"Step {self.state.global_step}: Task performances = {dict(state['task_performances'])}")
    print(f"Task counts = {dict(state['task_counts'])}")
    print(f"Current probs = {state['current_probs']}")
```

This will show if performances are being tracked and probabilities are adapting.