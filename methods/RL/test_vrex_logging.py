#!/usr/bin/env python3

"""
Quick test to verify VREx logging works properly after the fix.
This simulates the VREx scheduler logging to check for any errors.
"""

import sys
import os
sys.path.append('/data/shurui.gui/Projects/gateway/Sys2Bench/methods/RL')

from schedulers.variance_regularized_scheduler import _variance_regularized_schedule
import numpy as np

# Mock trainer class for testing
class MockTrainer:
    def __init__(self):
        self.accelerator = MockAccelerator()
        self.state = MockState()
    
    def log(self, metrics):
        print(f"[MOCK] trainer.log called with: {list(metrics.keys())}")

class MockAccelerator:
    def __init__(self):
        self.is_main_process = True

class MockState:
    def __init__(self):
        self.global_step = 42

def test_vrex_logging():
    """Test VREx scheduler logging functionality"""
    print("Testing VREx logging functionality...")
    
    # Create mock trainer
    trainer = MockTrainer()
    
    # Initialize scheduler state
    num_tasks = 4
    total_iterations = 100
    batch_size = 8
    
    # Simulate some training steps
    for step in range(5):
        print(f"\n--- Test Step {step + 1} ---")
        
        # Call the scheduler
        try:
            task_indices = _variance_regularized_schedule(
                trainer, step, num_tasks, total_iterations, 
                batch_size, scheduler_params={'beta': 0.7}
            )
            print(f"VREx scheduler returned task indices: {task_indices}")
            
        except Exception as e:
            print(f"❌ Error in VREx scheduler: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n✅ VREx logging test completed!")

if __name__ == "__main__":
    test_vrex_logging()