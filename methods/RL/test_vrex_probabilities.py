#!/usr/bin/env python3
"""Test script to debug VREx scheduler probability calculations."""

import numpy as np
import math
from schedulers.variance_regularized_scheduler import _variance_regularized_schedule, update_variance_regularized_performance_v2, reset_variance_regularized_state

def test_vrex_probabilities():
    """Test the VREx scheduler with synthetic performance data."""
    
    # Reset any existing state
    reset_variance_regularized_state()
    
    # Parameters
    num_tasks = 4
    T = 1600
    
    # Simulate some training steps with different task performances
    print("=== Testing VREx Scheduler Probability Calculation ===\n")
    
    # Warmup phase (first 100 steps) - should be uniform
    print("1. Testing warmup phase (should be uniform):")
    probs = _variance_regularized_schedule(50, T, num_tasks, 
                                          min_prob=0.1, beta=0.7, 
                                          progression_bias=0.3,
                                          performance_threshold=0.6)
    print(f"   Step 50: {probs}")
    
    # Simulate performance updates after warmup
    print("\n2. Adding synthetic performance data:")
    # Task 0 (easiest): high performance
    for _ in range(50):
        update_variance_regularized_performance_v2([0], [0.8])
    print("   Task 0: 50 samples, avg reward ~0.8")
    
    # Task 1: medium-high performance  
    for _ in range(40):
        update_variance_regularized_performance_v2([1], [0.6])
    print("   Task 1: 40 samples, avg reward ~0.6")
    
    # Task 2: medium performance
    for _ in range(30):
        update_variance_regularized_performance_v2([2], [0.4])
    print("   Task 2: 30 samples, avg reward ~0.4")
    
    # Task 3 (hardest): low performance
    for _ in range(20):
        update_variance_regularized_performance_v2([3], [0.2])
    print("   Task 3: 20 samples, avg reward ~0.2")
    
    # Check probabilities at different steps
    print("\n3. Testing probability evolution:")
    test_steps = [100, 200, 400, 800, 1200, 1500]
    
    for step in test_steps:
        probs = _variance_regularized_schedule(step, T, num_tasks,
                                             min_prob=0.1, beta=0.7,
                                             progression_bias=0.3,
                                             performance_threshold=0.6)
        print(f"\n   Step {step}:")
        print(f"   Probabilities: {probs}")
        
        # Check if probabilities are changing
        probs_list = list(probs.values())
        if len(set(probs_list)) == 1:
            print("   WARNING: All probabilities are equal!")
        else:
            print(f"   Min prob: {min(probs_list):.4f}, Max prob: {max(probs_list):.4f}")
    
    # Detailed debug at step 400
    print("\n4. Detailed debug at step 400:")
    # Enable debug prints by calling it at step 400
    probs = _variance_regularized_schedule(400, T, num_tasks,
                                         min_prob=0.1, beta=0.7,
                                         progression_bias=0.3,
                                         performance_threshold=0.6)
    
    # Check internal state
    if hasattr(_variance_regularized_schedule, 'state'):
        state = _variance_regularized_schedule.state
        print(f"\n   Internal state inspection:")
        print(f"   Group weights: {state['group_weights']}")
        print(f"   Task mastery: {state['task_mastery']}")
        print(f"   Task counts: {dict(state['task_counts'])}")
        
        # Calculate means manually
        for i in range(num_tasks):
            perfs = list(state['task_performances'][i])
            if perfs:
                print(f"   Task {i} mean reward: {np.mean(perfs):.4f}")

if __name__ == "__main__":
    test_vrex_probabilities()