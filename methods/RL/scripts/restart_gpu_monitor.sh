#!/bin/bash
# Script to restart GPU monitor with occupation feature

echo "Restarting GPU monitor with automatic occupation..."

# Kill existing monitor if running
if [ -f "gpu_monitor.pid" ]; then
    OLD_PID=$(cat gpu_monitor.pid)
    echo "Killing old monitor (PID: $OLD_PID)..."
    kill $OLD_PID 2>/dev/null
    rm gpu_monitor.pid
fi

# Kill any existing preparation processes
if [ -f "rl_prep_pids.txt" ]; then
    echo "Cleaning up existing RL preparation processes..."
    cat rl_prep_pids.txt | xargs kill 2>/dev/null
    rm rl_prep_pids.txt
fi

# Start new RL environment monitor in background
echo "Starting RL environment monitor..."
cd /data/shurui.gui/Projects/Sys2Bench
nohup python methods/RL/monitoring/rl_environment_monitor.py > rl_env_monitor.log 2>&1 &
NEW_PID=$!

# Save PID
echo $NEW_PID > gpu_monitor.pid
echo "✓ RL environment monitor started with PID: $NEW_PID"
echo "  Log file: rl_env_monitor.log"
echo ""
echo "The monitor will:"
echo "1. Check system resources every 5 minutes"
echo "2. Prepare RL training environment when resources are available"
echo "3. Send notification when environment is ready"
echo "4. Maintain environment readiness until training begins"
echo ""
echo "To check status: tail -f rl_env_monitor.log"
echo "To stop monitor: kill $(cat gpu_monitor.pid)"