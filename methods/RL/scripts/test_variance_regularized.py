"""
Test script for variance regularized scheduler
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

import numpy as np
from variance_regularized_scheduler import _variance_regularized_schedule, update_variance_regularized_performance, reset_variance_regularized_state
import matplotlib.pyplot as plt

def test_variance_regularized_scheduler():
    """Test the variance regularized scheduler behavior"""
    
    # Reset state
    reset_variance_regularized_state()
    
    # Test parameters
    num_tasks = 4
    total_iterations = 1000
    
    # Scheduler parameters
    scheduler_params = {
        'window_size': 50,
        'min_prob': 0.1,
        'temperature': 1.0,
        'beta': 0.7,
        'warmup_steps': 100,
        'vrex_penalty_weight': 1.0,
        'groupdro_alpha': 0.01
    }
    
    # Track probabilities over time
    prob_history = {i: [] for i in range(num_tasks)}
    
    # Simulate different task performances
    task_base_performance = [0.8, 0.6, 0.4, 0.2]  # Task 0 is easiest, Task 3 is hardest
    task_variance = [0.1, 0.2, 0.3, 0.4]  # Higher tasks have more variance
    
    for t in range(total_iterations):
        # Get current probabilities
        probs = _variance_regularized_schedule(t, total_iterations, num_tasks, **scheduler_params)
        
        # Record probabilities
        for i in range(num_tasks):
            prob_history[i].append(probs[i])
        
        # Simulate performance feedback (every 10 steps)
        if t % 10 == 0 and t > 0:
            # Sample some tasks based on current probabilities
            sampled_tasks = np.random.choice(num_tasks, size=32, p=[probs[i] for i in range(num_tasks)])
            
            # Generate synthetic performance data
            task_ids = []
            performances = []
            
            for task in sampled_tasks:
                task_ids.append(task)
                # Performance with noise
                perf = task_base_performance[task] + np.random.normal(0, task_variance[task])
                perf = np.clip(perf, 0, 1)
                performances.append(perf)
            
            # Update scheduler with performance data
            update_variance_regularized_performance(task_ids, performances)
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Probability evolution
    plt.subplot(2, 2, 1)
    for i in range(num_tasks):
        plt.plot(prob_history[i], label=f'Task {i}')
    plt.xlabel('Iteration')
    plt.ylabel('Sampling Probability')
    plt.title('Task Sampling Probabilities Over Time')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Final probabilities
    plt.subplot(2, 2, 2)
    final_probs = [prob_history[i][-1] for i in range(num_tasks)]
    plt.bar(range(num_tasks), final_probs)
    plt.xlabel('Task')
    plt.ylabel('Final Probability')
    plt.title('Final Sampling Probabilities')
    plt.grid(True)
    
    # Plot 3: Expected vs actual performance
    plt.subplot(2, 2, 3)
    plt.bar(range(num_tasks), task_base_performance, alpha=0.5, label='Expected')
    plt.xlabel('Task')
    plt.ylabel('Performance')
    plt.title('Task Performance Levels')
    plt.legend()
    plt.grid(True)
    
    # Plot 4: Probability changes over time (moving average)
    plt.subplot(2, 2, 4)
    window = 50
    for i in range(num_tasks):
        smoothed = np.convolve(prob_history[i], np.ones(window)/window, mode='valid')
        plt.plot(smoothed, label=f'Task {i}')
    plt.xlabel('Iteration')
    plt.ylabel('Smoothed Probability')
    plt.title(f'Smoothed Probabilities (window={window})')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('variance_regularized_scheduler_test.png')
    print("Test plot saved as variance_regularized_scheduler_test.png")
    
    # Print summary statistics
    print("\nScheduler Test Summary:")
    print(f"Number of tasks: {num_tasks}")
    print(f"Total iterations: {total_iterations}")
    print(f"Task base performances: {task_base_performance}")
    print(f"Final probabilities: {[f'{p:.3f}' for p in final_probs]}")
    print(f"Expected behavior: Lower performing tasks should have higher final probabilities")
    
    # Verify that lower performing tasks get higher probabilities
    perf_rank = np.argsort(task_base_performance)
    prob_rank = np.argsort(final_probs)[::-1]  # Reverse for descending
    
    print(f"\nPerformance ranking (worst to best): {perf_rank}")
    print(f"Probability ranking (highest to lowest): {prob_rank}")
    
    if np.array_equal(perf_rank, prob_rank):
        print("✓ Scheduler correctly prioritizes lower-performing tasks!")
    else:
        print("⚠ Scheduler may not be prioritizing tasks as expected")

if __name__ == "__main__":
    test_variance_regularized_scheduler()