#!/bin/bash
# Script to restart GPU monitor in tmux session with immediate GPU preparation

TMUX_SESSION="gpu_monitor"
MONITOR_SCRIPT="methods/RL/monitoring/rl_environment_monitor_immediate.py"

echo "Restarting RL Environment Monitor in tmux session..."

# Kill existing monitor if running
if [ -f "gpu_monitor.pid" ]; then
    OLD_PID=$(cat gpu_monitor.pid)
    echo "Killing old monitor (PID: $OLD_PID)..."
    kill $OLD_PID 2>/dev/null
    rm -f gpu_monitor.pid
fi

# Kill any existing monitor processes
echo "Checking for any existing monitor processes..."
pkill -f "rl_environment_monitor" 2>/dev/null

# Clean up any existing preparation processes
if [ -f "rl_prep_pids.txt" ]; then
    echo "Cleaning up existing RL preparation processes..."
    cat rl_prep_pids.txt | awk '{print $1}' | xargs kill 2>/dev/null
    rm -f rl_prep_pids.txt
fi

# Kill existing tmux session if it exists
tmux kill-session -t $TMUX_SESSION 2>/dev/null
sleep 1

# Activate conda environment
source /data/shurui.gui/mambaforge/etc/profile.d/conda.sh
conda activate sys2bench

# Make scripts executable
chmod +x $MONITOR_SCRIPT
chmod +x methods/RL/monitoring/sys_rl_prepare.py

# Create new tmux session and run monitor
echo "Starting monitor in tmux session '$TMUX_SESSION'..."
tmux new-session -d -s $TMUX_SESSION -c /data/shurui.gui/Projects/Sys2Bench

# Send commands to tmux session
tmux send-keys -t $TMUX_SESSION "cd /data/shurui.gui/Projects/Sys2Bench" C-m
tmux send-keys -t $TMUX_SESSION "source /data/shurui.gui/mambaforge/etc/profile.d/conda.sh" C-m
tmux send-keys -t $TMUX_SESSION "conda activate sys2bench" C-m
tmux send-keys -t $TMUX_SESSION "python $MONITOR_SCRIPT" C-m

echo ""
echo "✓ RL Environment Monitor started in tmux session!"
echo ""
echo "The monitor will:"
echo "  • Check GPUs every 60 seconds"
echo "  • IMMEDIATELY prepare any GPU with 50GB+ free memory"
echo "  • Send email notifications when GPUs are prepared"
echo "  • Continue monitoring for additional GPUs"
echo ""
echo "To view real-time logs:"
echo "  tmux attach -t $TMUX_SESSION"
echo ""
echo "To detach from tmux:"
echo "  Press Ctrl+B, then D"
echo ""
echo "To stop the monitor:"
echo "  tmux kill-session -t $TMUX_SESSION"
echo ""
echo "Current GPU status:"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv