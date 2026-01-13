#!/bin/bash
# Script to check GPU monitor status on remote server

echo "Checking GPU monitor status on remote server..."
echo "============================================="

ssh shurui.gui@dive7.engr.tamu.edu << 'EOF'
cd /data/shurui.gui/Projects/Sys2Bench

# Check if monitor is running
if [ -f "gpu_monitor.pid" ]; then
    PID=$(cat gpu_monitor.pid)
    echo "Monitor PID: $PID"
    
    if ps -p $PID > /dev/null 2>&1; then
        echo "Status: ✓ Running"
        
        # Get process info
        echo ""
        echo "Process info:"
        ps -fp $PID
    else
        echo "Status: ✗ Not running (PID file exists but process is dead)"
    fi
else
    echo "Status: No PID file found"
fi

# Check for preparation processes
echo ""
echo "RL Preparation processes:"
if [ -f "rl_prep_pids.txt" ]; then
    echo "Preparation PIDs listed in file:"
    cat rl_prep_pids.txt
    echo ""
    echo "Active preparation processes:"
    ps aux | grep sys_rl_prepare.py | grep -v grep || echo "None found"
else
    echo "No preparation PID file found"
fi

# Show log file
echo ""
echo "=== Latest log entries ==="
if [ -f "rl_env_monitor.log" ]; then
    tail -50 rl_env_monitor.log
else
    echo "Log file not found"
fi

# Current GPU status
echo ""
echo "=== Current GPU status ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total --format=csv

EOF