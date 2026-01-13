#!/usr/bin/env python3

import wandb
import pandas as pd

def check_wandb_run(run_id="snn90u6m", project="dive-ci/Sys2Bench"):
    """Check specific WandB run for available metrics and VREx curves"""
    
    # Initialize wandb API
    api = wandb.Api()
    
    # Get the specific run
    try:
        run = api.run(f"{project}/{run_id}")
        print(f"✅ Found run: {run.name}")
        print(f"   State: {run.state}")
        print(f"   Created: {run.created_at}")
        print(f"   Config: {run.config.get('algorithm.training.curriculum_schedule', 'N/A')}")
        
        # Get all available metrics
        print("\n📊 Available Metrics:")
        all_metrics = set()
        vrex_metrics = set()
        
        # Scan history to get all metric keys
        history = run.scan_history()
        for i, row in enumerate(history):
            if i == 0:  # First row to see initial metrics
                print(f"\n   First step metrics:")
                for key, value in row.items():
                    if not key.startswith('_'):
                        print(f"      - {key}: {value}")
            
            for key in row.keys():
                if not key.startswith('_'):
                    all_metrics.add(key)
                    if 'vrex/' in key:
                        vrex_metrics.add(key)
            
            if i >= 100:  # Check first 100 steps
                break
        
        print(f"\n   Total unique metrics: {len(all_metrics)}")
        
        # Check for VREx metrics
        if vrex_metrics:
            print(f"\n✅ VREx Metrics Found ({len(vrex_metrics)}):")
            for metric in sorted(vrex_metrics):
                print(f"      - {metric}")
        else:
            print("\n❌ NO VREx-specific metrics found!")
        
        # Show all metrics
        print("\n📋 All metrics:")
        for metric in sorted(all_metrics):
            print(f"   - {metric}")
        
        # Get reward data - check different possible reward column names
        print("\n📈 Reward Curve Data:")
        history_df = run.history(samples=1000)
        
        # Check for different reward metric names
        reward_columns = [col for col in history_df.columns if 'reward' in col.lower() and not col.startswith('_')]
        print(f"   Found reward columns: {reward_columns}")
        
        # Use the countdown reward metric
        reward_col = 'train/rewards/_countdown_reward_fn'
        if reward_col in history_df.columns:
            reward_data = history_df[reward_col].dropna()
            print(f"\n   📊 {reward_col} data:")
            print(f"   - Total data points: {len(reward_data)}")
            if len(reward_data) > 0:
                print(f"   - First reward: {reward_data.iloc[0]:.4f}")
                print(f"   - Last reward: {reward_data.iloc[-1]:.4f}")
                print(f"   - Max reward: {reward_data.max():.4f}")
                print(f"   - Min reward: {reward_data.min():.4f}")
                print(f"   - Mean reward: {reward_data.mean():.4f}")
            else:
                print("   - No reward data available")
            
            # Show complete reward history
            print(f"\n   📈 Complete Reward History:")
            reward_list = []
            for idx, val in reward_data.items():
                reward_list.append(val)
                print(f"      Step {idx}: {val:.4f}")
            
            # Also show as a Python list for easy copying
            print(f"\n   📊 Reward values as list:")
            print(f"      {reward_list}")
            
            # Show reward improvement
            if len(reward_data) > 1:
                improvement = reward_data.iloc[-1] - reward_data.iloc[0]
                print(f"\n   📈 Improvement: {improvement:.4f} ({improvement/reward_data.iloc[0]*100:.1f}% increase)")
        
        # Also check train/reward
        if 'train/reward' in history_df.columns:
            reward_data = history_df['train/reward'].dropna()
            print(f"\n   📊 train/reward data:")
            print(f"   - Total data points: {len(reward_data)}")
            if len(reward_data) > 0:
                print(f"   - Last 5 values: {reward_data.tail(5).values}")
            
        # Check summary
        print("\n📊 Run Summary:")
        for key, value in run.summary.items():
            if not key.startswith('_') and 'vrex' in key.lower():
                print(f"   - {key}: {value}")
        
    except Exception as e:
        print(f"❌ Error accessing run: {e}")

if __name__ == "__main__":
    check_wandb_run()