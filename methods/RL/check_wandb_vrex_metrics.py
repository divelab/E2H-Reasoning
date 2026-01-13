#!/usr/bin/env python3

import wandb
import sys
from datetime import datetime, timedelta

def check_vrex_metrics():
    """Check if VREx-specific metrics are being logged to WandB"""
    
    try:
        # Initialize wandb API
        api = wandb.Api()
        
        # Get recent runs from the Sys2Bench project
        runs = api.runs("dive-ci/Sys2Bench")
        
        print("=== Recent WandB Runs ===")
        
        vrex_runs = []
        # Check first 20 runs for efficiency
        for i, run in enumerate(runs):
            if i >= 20:
                break
            if "variance_regularized" in run.name or "vrex" in run.name.lower():
                vrex_runs.append(run)
                print(f"\n🔍 VREx Run: {run.name}")
                print(f"   ID: {run.id}")
                print(f"   State: {run.state}")
                print(f"   Created: {run.created_at}")
                print(f"   Config: curriculum_schedule = {run.config.get('algorithm.training.curriculum_schedule', 'N/A')}")
                
                # Check for VREx-specific metrics
                vrex_metrics = []
                all_metrics = set()
                
                try:
                    # Get a sample of metrics from the run
                    history = run.scan_history(keys=None, page_size=10)
                    for i, row in enumerate(history):
                        if i > 10:  # Limit to avoid too much data
                            break
                        for key in row.keys():
                            all_metrics.add(key)
                            if key.startswith('vrex/'):
                                vrex_metrics.append(key)
                except Exception as e:
                    print(f"   Error reading history: {e}")
                    continue
                
                print(f"   📊 Total Metrics: {len(all_metrics)}")
                
                if vrex_metrics:
                    print(f"   ✅ VREx Metrics Found ({len(vrex_metrics)}):")
                    for metric in sorted(set(vrex_metrics)):
                        print(f"      - {metric}")
                else:
                    print("   ❌ NO VREx-specific metrics found!")
                    print("   📋 All available metrics:")
                    for metric in sorted(list(all_metrics)[:20]):  # Show first 20
                        print(f"      - {metric}")
                    if len(all_metrics) > 20:
                        print(f"      ... and {len(all_metrics) - 20} more")
        
        if not vrex_runs:
            print("❌ No VREx runs found in recent runs!")
            print("\n📋 All recent runs:")
            run_list = list(runs)[:10]
            for run in run_list:
                print(f"   - {run.name} (state: {run.state})")
        
        return len(vrex_runs), sum(1 for run in vrex_runs if any(key.startswith('vrex/') for key in run.summary.keys()))
        
    except Exception as e:
        print(f"❌ Error accessing WandB: {e}")
        return 0, 0

if __name__ == "__main__":
    print("Checking VREx metrics in WandB...")
    num_vrex_runs, num_with_metrics = check_vrex_metrics()
    print(f"\n📊 Summary: {num_vrex_runs} VREx runs found, {num_with_metrics} with VREx metrics")