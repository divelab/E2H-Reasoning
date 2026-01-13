#!/bin/bash
# Script to remotely restart monitor in tmux with immediate GPU preparation

echo "Connecting to remote server to restart monitor in tmux..."
echo "=" * 60

ssh shurui.gui@dive7.engr.tamu.edu << 'EOF'
cd /data/shurui.gui/Projects/Sys2Bench

# First, let's see current GPU status
echo "Current GPU Status:"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv
echo ""

# Check for existing monitor
if [ -f "gpu_monitor.pid" ]; then
    OLD_PID=$(cat gpu_monitor.pid)
    if ps -p $OLD_PID > /dev/null 2>&1; then
        echo "Found running monitor (PID: $OLD_PID), killing it..."
        kill $OLD_PID
        sleep 2
    fi
    rm -f gpu_monitor.pid
fi

# Kill any monitor processes
pkill -f "rl_environment_monitor" 2>/dev/null

# Clean up preparation processes
if [ -f "rl_prep_pids.txt" ]; then
    echo "Cleaning up existing preparation processes..."
    cat rl_prep_pids.txt | awk '{print $1}' | xargs kill 2>/dev/null
    rm -f rl_prep_pids.txt
fi

# Make scripts executable
chmod +x methods/RL/scripts/restart_monitor_tmux.sh
chmod +x methods/RL/monitoring/rl_environment_monitor_immediate.py
chmod +x methods/RL/monitoring/sys_rl_prepare.py

# Run the tmux restart script
bash methods/RL/scripts/restart_monitor_tmux.sh

echo ""
echo "Monitor should now be running in tmux!"
echo ""
echo "To check the monitor from your local machine:"
echo "  ssh shurui.gui@dive7.engr.tamu.edu 'tmux attach -t gpu_monitor'"
echo ""
echo "Or SSH in and run:"
echo "  tmux attach -t gpu_monitor"

EOF

echo ""
echo "Remote operation complete!"