#!/bin/bash
# Simple script to run a command in tmux session
# Usage: ./run_in_tmux.sh <command> [session_name] [window_index]
#
# Examples:
#   ./run_in_tmux.sh "echo hello"                    # Run in session 'claude', window 0
#   ./run_in_tmux.sh "python script.py" claude 1     # Run in session 'claude', window 1
#   ./run_in_tmux.sh C-c claude                      # Send Ctrl+C to session 'claude'
#
# Note: The default shell in tmux sessions is fish, not bash
# For running monitor: ./run_in_tmux.sh "python methods/RL/monitoring/rl_environment_monitor_immediate.py"

if [ $# -eq 0 ]; then
    echo "Usage: $0 <command> [session_name] [window_index]"
    exit 1
fi

COMMAND="$1"
SESSION="${2:-claude}"
WINDOW="${3:-0}"

echo "=== Sending command to tmux session: $SESSION, window: $WINDOW ==="
echo "Command: $COMMAND"

ssh shurui.gui@dive7.engr.tamu.edu << EOF
# Need conda environment for tmux command
source /data/shurui.gui/mambaforge/etc/profile.d/conda.sh
conda activate sys2bench

# Check if session exists
if ! tmux has-session -t $SESSION 2>/dev/null; then
    echo "Creating new tmux session '$SESSION'..."
    tmux new-session -d -s $SESSION -c /data/shurui.gui/Projects/Sys2Bench
    sleep 1
fi

# Send the command
# Handle special keys like C-c (Ctrl+C)
if [ "$COMMAND" = "C-c" ] || [ "$COMMAND" = "C-z" ]; then
    if [ "$WINDOW" = "0" ]; then
        tmux send-keys -t $SESSION $COMMAND
    else
        tmux send-keys -t $SESSION:$WINDOW $COMMAND
    fi
else
    # Regular commands need Enter
    if [ "$WINDOW" = "0" ]; then
        tmux send-keys -t $SESSION "$COMMAND" Enter
    else
        tmux send-keys -t $SESSION:$WINDOW "$COMMAND" Enter
    fi
fi

echo "Command sent successfully"
EOF