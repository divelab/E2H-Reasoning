#!/bin/bash
# Simple script to check tmux session content
# Usage: ./check_tmux.sh [session_name] [window_index]
# 
# Examples:
#   ./check_tmux.sh claude          # Check session 'claude', window 0 (default)
#   ./check_tmux.sh claude 1        # Check session 'claude', window 1
#   ./check_tmux.sh                 # Default: session 'claude', window 0

SESSION="${1:-claude}"
WINDOW="${2:-0}"

echo "=== Checking tmux session: $SESSION, window: $WINDOW ==="

ssh shurui.gui@dive7.engr.tamu.edu << EOF
# Need conda environment for tmux command
source /data/shurui.gui/mambaforge/etc/profile.d/conda.sh
conda activate sys2bench

# Check if session exists
if ! tmux has-session -t $SESSION 2>/dev/null; then
    echo "No tmux session '$SESSION' found"
    exit 1
fi

# Capture pane content (last 100 lines should be enough)
# If window is not specified, just use session name
if [ "$WINDOW" = "0" ]; then
    tmux capture-pane -t $SESSION -p 2>&1 | grep -v "no version information" | tail -100
else
    tmux capture-pane -t $SESSION:$WINDOW -p 2>&1 | grep -v "no version information" | tail -100
fi
EOF