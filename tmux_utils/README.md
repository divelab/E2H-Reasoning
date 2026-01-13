# TMUX Utilities for Remote Server Management

## Overview
These utilities help manage tmux sessions on the remote server (dive7.engr.tamu.edu). They handle the specific environment setup required for our server, including conda activation and fish shell compatibility.

## Important Notes
1. **Default Shell**: The remote server uses `fish` shell in tmux sessions
2. **Conda**: In fish shell, use `conda activate <env>` directly (no need to source conda.sh)
3. **Library Warnings**: You'll see warnings about `libtinfo.so.6` - these are harmless and can be ignored

## Scripts

### check_tmux.sh
Check the content of any tmux session.

**Usage:**
```bash
./check_tmux.sh [session_name] [window_index]
```

**Examples:**
```bash
./check_tmux.sh                    # Check default session 'claude', window 0
./check_tmux.sh claude             # Check session 'claude', window 0
./check_tmux.sh mysession         # Check session 'mysession', window 0
./check_tmux.sh claude 1           # Check session 'claude', window 1
```

**Note**: Shows the last 100 lines of the tmux pane content.

### run_in_tmux.sh
Run any command in a tmux session.

**Usage:**
```bash
./run_in_tmux.sh <command> [session_name] [window_index]
```

**Examples:**
```bash
# Run simple commands
./run_in_tmux.sh "echo hello"                    # Run in default session 'claude'
./run_in_tmux.sh "ls -la" mysession             # Run in session 'mysession'

# Send control keys
./run_in_tmux.sh C-c claude                      # Send Ctrl+C to stop a process
./run_in_tmux.sh C-z claude                      # Send Ctrl+Z to suspend a process

# Run Python scripts
./run_in_tmux.sh "python my_script.py" claude

# Activate conda environment (fish shell)
./run_in_tmux.sh "conda activate sys2bench" claude
```

**Note**: The script automatically creates the session if it doesn't exist.

## Common Workflows

### 1. Start a Long-Running Process
```bash
# Start the process
./run_in_tmux.sh "python methods/RL/monitoring/rl_environment_monitor_immediate.py" claude

# Check its output
./check_tmux.sh claude

# Stop it when needed
./run_in_tmux.sh C-c claude
```

### 2. Run Multiple Commands
```bash
./run_in_tmux.sh "cd /data/shurui.gui/Projects/Sys2Bench" claude
./run_in_tmux.sh "conda activate sys2bench" claude
./run_in_tmux.sh "python my_script.py" claude
```

### 3. Debug a Session
```bash
# Check what's running
./check_tmux.sh claude

# If something is stuck, send Ctrl+C
./run_in_tmux.sh C-c claude

# Check again
./check_tmux.sh claude
```

## Tips
- Always check session content before running new commands
- Use meaningful session names for different tasks
- Remember that fish shell is the default in tmux sessions
- For interactive work, attach directly: `tmux attach -t session_name`

## Troubleshooting
- **"No tmux session found"**: The session doesn't exist. run_in_tmux.sh will create it automatically
- **Can't see output**: Use `head` not `tail` when viewing tmux content (panes can be large)
- **Commands not working**: Check if you need to activate conda environment first