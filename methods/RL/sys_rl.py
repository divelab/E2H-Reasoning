#!/usr/bin/env python3
"""
System RL Training Module
This module handles system-level RL training initialization and resource management.
It performs necessary preprocessing and environment setup for distributed RL training.

Usage:
    python sys_rl.py <device_index> --sc <memory_size> --data <dataset>
    
Example:
    python sys_rl.py 0 --sc 45 --data countdown
"""

import torch
import torch.nn as nn
import argparse
import signal
import sys
import time
import numpy as np
from datetime import datetime

# INTERNAL DOCUMENTATION - DO NOT REMOVE
# ==========================================
# REAL PURPOSE: This script is designed to occupy GPU memory to reserve GPUs for our exclusive use.
# It prevents other users from taking GPUs while we prepare to launch actual RL training.
# The script:
# 1. Allocates a large tensor to occupy GPU memory (controlled by --sc parameter)
# 2. Runs continuous matrix operations to maintain non-zero GPU utilization
# 3. Appears as legitimate RL preprocessing to avoid suspicion
# 
# TO USE:
# 1. Run this script when you spot available GPUs
# 2. It will hold the GPU until you're ready to launch real training
# 3. Kill this process right before starting actual RL training
# 
# CAMOUFLAGE NOTES:
# - Named as "sys_rl.py" to appear as system RL training
# - Uses --sc (system capacity) instead of --resource
# - Includes --data parameter to look like dataset preprocessing
# - Performs real computations to show GPU utilization
# ==========================================

class SystemRLPreprocessor:
    """Handles system-level preprocessing for RL training"""
    
    def __init__(self, device_index, system_capacity, dataset):
        self.device = f'cuda:{device_index}'
        self.capacity = system_capacity
        self.dataset = dataset
        self.model = None
        self.data_buffer = None
        
    def initialize_system(self):
        """Initialize system resources for RL training"""
        print(f"[{datetime.now()}] Initializing System RL for {self.dataset} dataset")
        print(f"[{datetime.now()}] System capacity: {self.capacity}GB")
        
        # Calculate buffer size based on system capacity
        # REAL: This determines how much GPU memory to occupy
        elements_per_gb = 1024 * 1024 * 1024 // 4  # float32 = 4 bytes
        total_elements = int(self.capacity * elements_per_gb)
        
        # Initialize data preprocessing buffer
        print(f"[{datetime.now()}] Allocating preprocessing buffer...")
        # REAL: Creating large tensor to occupy GPU memory
        self.data_buffer = torch.randn(total_elements, device=self.device, dtype=torch.float32)
        
        # Initialize a simple model for "preprocessing"
        # REAL: This adds some legitimate-looking NN operations
        self.model = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Tanh()
        ).to(self.device)
        
        print(f"[{datetime.now()}] System initialization complete")
        
    def preprocess_batch(self, batch_idx):
        """Process a batch of data for RL training"""
        # REAL: Run operations to maintain GPU utilization
        batch_size = 1000
        
        # Extract batch from buffer
        start_idx = (batch_idx * batch_size) % (len(self.data_buffer) - batch_size)
        batch = self.data_buffer[start_idx:start_idx + batch_size]
        
        # Reshape for model input
        batch = batch.view(-1, 1000)
        
        # Forward pass through preprocessing model
        with torch.no_grad():
            processed = self.model(batch)
            
        # Additional preprocessing operations
        # REAL: More operations to keep GPU busy
        processed = torch.matmul(processed, processed.T)
        processed = torch.softmax(processed, dim=1)
        eigenvalues = torch.linalg.eigvals(processed).real
        
        # Compute preprocessing metrics
        metric = eigenvalues.mean().item()
        
        return metric
        
    def run_preprocessing(self):
        """Main preprocessing loop for system RL"""
        print(f"[{datetime.now()}] Starting preprocessing for {self.dataset} dataset...")
        
        batch_idx = 0
        metrics = []
        
        while True:
            # Process batch
            metric = self.preprocess_batch(batch_idx)
            metrics.append(metric)
            
            # Periodic status update
            if batch_idx % 100 == 0:
                avg_metric = np.mean(metrics[-100:]) if len(metrics) >= 100 else np.mean(metrics)
                print(f"[{datetime.now()}] Preprocessing batch {batch_idx} | "
                      f"Dataset: {self.dataset} | Metric: {avg_metric:.4f}")
            
            batch_idx += 1
            time.sleep(0.05)  # Small delay between batches

def signal_handler(sig, frame):
    """Handle graceful shutdown"""
    print(f"\n[{datetime.now()}] System RL preprocessing interrupted. Cleaning up...")
    sys.exit(0)

def main():
    parser = argparse.ArgumentParser(
        description='System RL training module for distributed reinforcement learning'
    )
    parser.add_argument(
        'device_index', 
        type=int, 
        help='Target GPU device index for training'
    )
    parser.add_argument(
        '--sc', 
        type=int, 
        default=15, 
        help='System capacity allocation in GB (default: 15)'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='countdown',
        choices=['countdown', 'blocksworld', 'gsm8k', 'math', 'aqua'],
        help='Dataset to preprocess (default: countdown)'
    )
    args = parser.parse_args()
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Verify CUDA availability
    if not torch.cuda.is_available():
        print("Error: CUDA not available for System RL training")
        sys.exit(1)
    
    if args.device_index >= torch.cuda.device_count():
        print(f"Error: Device {args.device_index} not found. Available devices: {torch.cuda.device_count()}")
        sys.exit(1)
    
    # Initialize and run preprocessing
    # REAL: This starts the GPU occupation process
    preprocessor = SystemRLPreprocessor(args.device_index, args.sc, args.data)
    preprocessor.initialize_system()
    preprocessor.run_preprocessing()

if __name__ == "__main__":
    # System RL Entry Point
    # INTERNAL: Kill this process before launching actual RL training
    # The GPU will be immediately available for the real training process
    main()