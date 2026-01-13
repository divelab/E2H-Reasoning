#!/usr/bin/env python3
"""Analyze WandB reward data in detail"""

import wandb
import pandas as pd
import numpy as np

api = wandb.Api()

run_ids = {
    "Config 1 (Default Enhanced)": "j6q90bp2",
    "Config 2 (Aggressive)": "gx4lv0x2", 
    "Config 3 (Conservative)": "y8cs16pa"
}

print("Detailed Reward Analysis")
print("="*80)

for config_name, run_id in run_ids.items():
    print(f"\n{config_name} (Run: {run_id})")
    print("-"*60)
    
    try:
        run = api.run(f"dive-ci/Sys2Bench/{run_id}")
        history = run.history()
        
        # Get the reward column - try different possible names
        reward_col = None
        for col in ['train/reward', 'reward', 'train/rewards/_countdown_reward_fn']:
            if col in history.columns:
                reward_col = col
                break
        
        if reward_col:
            rewards = history[reward_col].dropna()
            steps = rewards.index
            
            print(f"Using reward column: {reward_col}")
            print(f"Total logged steps: {len(rewards)}")
            
            # Show progression
            print("\nReward Progression:")
            checkpoints = [0, 10, 20, 30, 40, 50, 100, 150, 200, 250, 300, 350, 390]
            for step in checkpoints:
                if step < len(rewards):
                    print(f"  Step {step:3d}: {rewards.iloc[step]:.4f}")
                    
            # Final statistics
            print(f"\nFinal 20 steps:")
            print(f"  Mean: {rewards.iloc[-20:].mean():.4f}")
            print(f"  Std:  {rewards.iloc[-20:].std():.4f}")
            print(f"  Min:  {rewards.iloc[-20:].min():.4f}")
            print(f"  Max:  {rewards.iloc[-20:].max():.4f}")
            
            # Check completion lengths
            if 'train/completion_length' in history.columns:
                lengths = history['train/completion_length'].dropna()
                print(f"\nCompletion Lengths:")
                print(f"  Initial (first 20): {lengths.iloc[:20].mean():.1f}")
                print(f"  Final (last 20): {lengths.iloc[-20:].mean():.1f}")
                
        else:
            print("No reward column found!")
            print(f"Available columns: {list(history.columns)[:10]}...")
            
    except Exception as e:
        print(f"Error: {e}")

print("\n" + "="*80)
print("COMPARISON SUMMARY")
print("="*80)

# Collect final rewards for comparison
final_rewards = {}
for config_name, run_id in run_ids.items():
    try:
        run = api.run(f"dive-ci/Sys2Bench/{run_id}")
        history = run.history()
        
        for col in ['train/reward', 'reward', 'train/rewards/_countdown_reward_fn']:
            if col in history.columns:
                rewards = history[col].dropna()
                final_rewards[config_name] = rewards.iloc[-20:].mean()
                break
    except:
        final_rewards[config_name] = None

print("\nFinal Training Rewards (last 20 steps average):")
for config, reward in final_rewards.items():
    if reward is not None:
        print(f"  {config}: {reward:.4f}")
    else:
        print(f"  {config}: N/A")

# Check why VREx metrics aren't logged
print("\n" + "="*80)
print("DIAGNOSTIC: Why VREx metrics aren't logged")
print("="*80)
print("\nThe VREx scheduler includes logging code, but it's not appearing in WandB.")
print("Possible reasons:")
print("1. The update_variance_regularized_performance function may not be called with trainer")
print("2. The trainer.log() method might need different formatting")
print("3. The logging might be happening at the wrong time in the training loop")