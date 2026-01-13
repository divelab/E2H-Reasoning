#!/bin/bash
# Script to remotely kill old monitor and start new one

echo "Connecting to remote server to restart GPU monitor..."

ssh shurui.gui@dive7.engr.tamu.edu << 'EOF'
cd /data/shurui.gui/Projects/Sys2Bench

echo "=== Checking for existing monitors ==="

# Check for PID file
if [ -f "gpu_monitor.pid" ]; then
    OLD_PID=$(cat gpu_monitor.pid)
    echo "Found monitor PID file: $OLD_PID"
    
    # Check if process is still running
    if ps -p $OLD_PID > /dev/null 2>&1; then
        echo "Killing old monitor (PID: $OLD_PID)..."
        kill $OLD_PID
        sleep 2
    else
        echo "Old monitor already dead"
    fi
    rm -f gpu_monitor.pid
else
    echo "No PID file found"
fi

# Also check for any gpu_monitor_email.py processes
echo ""
echo "Checking for any gpu_monitor_email.py processes..."
MONITOR_PIDS=$(ps aux | grep "gpu_monitor_email.py" | grep -v grep | awk '{print $2}')
if [ ! -z "$MONITOR_PIDS" ]; then
    echo "Found monitor processes: $MONITOR_PIDS"
    echo "$MONITOR_PIDS" | xargs kill 2>/dev/null
    echo "Killed monitor processes"
else
    echo "No gpu_monitor_email.py processes found"
fi

# Clean up any existing preparation processes
if [ -f "rl_prep_pids.txt" ]; then
    echo ""
    echo "Cleaning up existing RL preparation processes..."
    cat rl_prep_pids.txt | xargs kill 2>/dev/null
    rm -f rl_prep_pids.txt
fi

# Start the new monitor
echo ""
echo "=== Starting RL environment monitor ==="
source /data/shurui.gui/mambaforge/etc/profile.d/conda.sh
conda activate sys2bench

# Make scripts executable
chmod +x methods/RL/monitoring/sys_rl_prepare.py
chmod +x methods/RL/monitoring/rl_environment_monitor.py
chmod +x methods/RL/scripts/restart_gpu_monitor.sh

# Run the restart script
bash methods/RL/scripts/restart_gpu_monitor.sh

# Check if it started successfully
sleep 3
if [ -f "gpu_monitor.pid" ]; then
    NEW_PID=$(cat gpu_monitor.pid)
    if ps -p $NEW_PID > /dev/null 2>&1; then
        echo ""
        echo "✓ SUCCESS: New monitor is running with PID: $NEW_PID"
        echo ""
        echo "Checking first few lines of log..."
        head -20 rl_env_monitor.log
    else
        echo "✗ ERROR: Monitor failed to start"
        tail -20 rl_env_monitor.log
    fi
else
    echo "✗ ERROR: PID file not created"
fi

echo ""
echo "=== Current GPU status ==="
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv

EOF

echo ""
echo "Remote operation complete!"