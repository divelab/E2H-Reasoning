#!/usr/bin/env python3
"""Test script to verify variance regularized scheduler integration"""

import numpy as np
from schedulers.variance_regularized_scheduler import (
    _variance_regularized_schedule,
    update_variance_regularized_performance,
    reset_variance_regularized_state
)

def test_basic_functionality():
    """Test basic variance regularized scheduler functionality"""
    print("Testing variance regularized scheduler...")
    
    # Reset state
    reset_variance_regularized_state()
    
    # Test parameters
    num_tasks = 4
    total_iterations = 100
    scheduler_params = {
        'vrex_lambda': 1.0,
        'group_lambda': 0.5,
        'alpha': 0.9,
        'window_size': 20,
        'epsilon': 0.01,
        'min_prob': True,
        'update_frequency': 5
    }
    
    # Create a scheduler function with parameters
    scheduler = lambda t: _variance_regularized_schedule(
        t, total_iterations, num_tasks, **scheduler_params
    )
    
    # Simulate training iterations
    print("\nSimulating training iterations...")
    for iteration in range(0, 50, 5):
        # Get current probabilities
        probs = scheduler(iteration)
        print(f"\nIteration {iteration}:")
        print(f"Task probabilities: {[f'{probs[i]:.3f}' for i in range(num_tasks)]}")
        
        # Simulate performance updates (task 0 performs poorly, task 3 performs well)
        if iteration > 0 and iteration % scheduler_params['update_frequency'] == 0:
            # Simulate a batch of 8 samples
            task_ids = np.random.choice(num_tasks, size=8, p=[probs[i] for i in range(num_tasks)])
            rewards = []
            for task_id in task_ids:
                if task_id == 0:
                    # Poor performance on task 0
                    reward = np.random.uniform(0.1, 0.3)
                elif task_id == 3:
                    # Good performance on task 3
                    reward = np.random.uniform(0.7, 0.9)
                else:
                    # Medium performance on other tasks
                    reward = np.random.uniform(0.4, 0.6)
                rewards.append(reward)
            
            # Update the scheduler
            update_variance_regularized_performance(task_ids.tolist(), rewards)
            
            # Check if state was updated
            if hasattr(_variance_regularized_schedule, 'state'):
                state = _variance_regularized_schedule.state
                print(f"Performance history lengths: {[len(state['task_performances'][i]) for i in range(num_tasks)]}")
                
                # Calculate and display recent averages
                for i in range(num_tasks):
                    if state['task_performances'][i]:
                        perf_list = list(state['task_performances'][i])
                        recent_perf = perf_list[-scheduler_params['window_size']:]
                        avg_perf = np.mean(recent_perf) if recent_perf else 0
                        print(f"Task {i} recent avg performance: {avg_perf:.3f}")

def test_integration_with_trainer():
    """Test integration pattern used in main.py"""
    print("\n\nTesting integration pattern...")
    
    # Simulate trainer with variance_regularized schedule
    class MockTrainer:
        def __init__(self):
            self.data_schedule = 'variance_regularized'
            self._last_batch_rewards = None
    
    class MockCountdownTrainer:
        def __init__(self):
            self.trainer = None
        
        def _countdown_reward_fn(self, completions, target, numbers, **kwargs):
            """Mock reward function"""
            rewards = [np.random.uniform(0, 1) for _ in completions]
            
            # Update variance regularized scheduler if we're in training mode
            if hasattr(self, 'trainer') and hasattr(self.trainer, 'data_schedule'):
                if self.trainer.data_schedule == 'variance_regularized':
                    # Store rewards in trainer for later use
                    self.trainer._last_batch_rewards = rewards
            
            return rewards
    
    # Create instances
    countdown_trainer = MockCountdownTrainer()
    mock_trainer = MockTrainer()
    countdown_trainer.trainer = mock_trainer
    
    # Test reward function
    completions = ["solution1", "solution2", "solution3"]
    target = [10, 20, 30]
    numbers = [[1,2,3], [4,5,6], [7,8,9]]
    
    rewards = countdown_trainer._countdown_reward_fn(completions, target, numbers)
    
    print(f"Rewards returned: {rewards}")
    print(f"Rewards stored in trainer: {mock_trainer._last_batch_rewards}")
    print(f"Storage successful: {mock_trainer._last_batch_rewards == rewards}")

if __name__ == "__main__":
    test_basic_functionality()
    test_integration_with_trainer()
    print("\n\nAll tests completed!")