#!/usr/bin/env python3
"""Analyze WandB runs for VREx experiments"""

import wandb
import pandas as pd
import numpy as np

# Initialize API
api = wandb.Api()

# Run IDs from the logs
run_ids = {
    "Config 1 (Default Enhanced)": "j6q90bp2",
    "Config 2 (Aggressive)": "gx4lv0x2", 
    "Config 3 (Conservative)": "y8cs16pa"
}

# Fetch runs
for config_name, run_id in run_ids.items():
    print(f"\n{'='*60}")
    print(f"Analyzing: {config_name}")
    print(f"Run ID: {run_id}")
    print(f"{'='*60}")
    
    try:
        # Get the run
        run = api.run(f"dive-ci/Sys2Bench/{run_id}")
        
        # Print run summary
        print(f"\nRun Name: {run.name}")
        print(f"State: {run.state}")
        print(f"Duration: {run.summary.get('_runtime', 'N/A')} seconds")
        
        # Get history dataframe
        history = run.history()
        
        # Check for reward metrics
        reward_cols = [col for col in history.columns if 'reward' in col.lower()]
        print(f"\nReward-related columns found: {reward_cols}")
        
        # Check for VREx-specific metrics
        vrex_cols = [col for col in history.columns if 'vrex' in col.lower()]
        print(f"VREx-specific columns found: {vrex_cols}")
        
        # Analyze main reward progression
        if 'reward' in history.columns:
            rewards = history['reward'].dropna()
            print(f"\nReward Statistics:")
            print(f"  - Initial reward (first 10 steps): {rewards.iloc[:10].mean():.4f}")
            print(f"  - Mid-training reward (steps 190-210): {rewards.iloc[190:210].mean():.4f}")
            print(f"  - Final reward (last 10 steps): {rewards.iloc[-10:].mean():.4f}")
            print(f"  - Overall mean: {rewards.mean():.4f}")
            print(f"  - Std deviation: {rewards.std():.4f}")
            print(f"  - Min: {rewards.min():.4f}, Max: {rewards.max():.4f}")
            
            # Check for reward progression
            print(f"\nReward progression (every 50 steps):")
            for i in range(0, len(rewards), 50):
                step_rewards = rewards.iloc[i:i+10]
                if len(step_rewards) > 0:
                    print(f"  Step {i}: {step_rewards.mean():.4f}")
        
        # Check completion lengths (indicator of reward hacking)
        if 'completion_length' in history.columns:
            lengths = history['completion_length'].dropna()
            print(f"\nCompletion Length Statistics:")
            print(f"  - Mean: {lengths.mean():.1f}")
            print(f"  - Std: {lengths.std():.1f}")
            print(f"  - Min: {lengths.min():.1f}, Max: {lengths.max():.1f}")
            
        # Check if any VREx metrics were logged
        if vrex_cols:
            print(f"\nVREx Metrics Found:")
            for col in vrex_cols[:5]:  # Show first 5
                values = history[col].dropna()
                if len(values) > 0:
                    print(f"  - {col}: mean={values.mean():.4f}, last={values.iloc[-1]:.4f}")
        else:
            print(f"\n⚠️  No VREx-specific metrics found in WandB logs!")
            
        # Task difficulty distribution (if logged)
        task_prob_cols = [col for col in history.columns if 'task_' in col and 'prob' in col]
        if task_prob_cols:
            print(f"\nTask Sampling Probabilities:")
            for col in task_prob_cols:
                values = history[col].dropna()
                if len(values) > 0:
                    print(f"  - {col}: last value = {values.iloc[-1]:.4f}")
                    
    except Exception as e:
        print(f"Error analyzing run: {e}")

print("\n\n" + "="*60)
print("SUMMARY")
print("="*60)

# Compare final rewards across configs
print("\nFinal Training Rewards:")
for config_name, run_id in run_ids.items():
    try:
        run = api.run(f"dive-ci/Sys2Bench/{run_id}")
        history = run.history()
        if 'reward' in history.columns:
            final_reward = history['reward'].dropna().iloc[-10:].mean()
            print(f"  - {config_name}: {final_reward:.4f}")
    except:
        print(f"  - {config_name}: Error retrieving data")

print("\n⚠️  Note: VREx-specific metrics (task probabilities, cross-task variance, etc.) do not appear to be logged in WandB.")
print("This suggests the scheduler's logging functionality may not be properly integrated with the trainer.")