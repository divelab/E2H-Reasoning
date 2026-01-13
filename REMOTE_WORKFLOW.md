# Remote Workflow Guide

This guide covers the essential workflow for developing locally and running code on remote servers. Follow these step-by-step examples to understand the process.

## Table of Contents

1. [🔄 The Basic Workflow](#-the-basic-workflow)
2. [📝 Step 1: Local Development](#-step-1-local-development)
3. [📤 Step 2: Sync to Remote](#-step-2-sync-to-remote)
4. [🖥️ Step 3: Run on Remote](#️-step-3-run-on-remote)
5. [📖 Complete Example: Running a Monitor](#-complete-example-running-a-monitor)
6. [🐛 Debugging TMUX Sessions](#-debugging-tmux-sessions)
7. [🐟 Fish Shell Notes](#-fish-shell-notes)
8. [🎯 Quick Reference](#-quick-reference)
9. [💡 Tips](#-tips)

## 🔄 The Basic Workflow

1. **Edit locally** → 2. **Sync to remote** → 3. **Run on remote** → 4. **Check results**

## 📝 Step 1: Local Development

Always edit files on your local machine:
```bash
# Edit your code locally
vim methods/RL/my_script.py
```

## 📤 Step 2: Sync to Remote

Use rsync to upload files:
```bash
# Single file
rsync -av my_script.py user@server:/path/to/destination/

# Multiple files
rsync -av file1.py file2.py user@server:/path/to/destination/

# Entire directory
rsync -av methods/RL/ user@server:/path/to/project/methods/RL/
```

**Example:**
```bash
rsync -av methods/RL/sys_rl.py shurui.gui@dive7.engr.tamu.edu:/data/shurui.gui/Projects/Sys2Bench/methods/RL/
```

## 🖥️ Step 3: Run on Remote

### Option A: Direct SSH (for quick commands)
```bash
ssh user@server "cd /path/to/project && python my_script.py"
```

### Option B: Using TMUX (for long-running processes)

We have tmux utilities in `tmux_utils/`:

**1. First, sync the utilities:**
```bash
rsync -av tmux_utils/ user@server:/path/to/project/tmux_utils/
```

**2. Run commands in tmux:**
```bash
cd tmux_utils

# Create session and run command
./run_in_tmux.sh "python my_script.py" session_name

# Check output
./check_tmux.sh session_name
```

## 📖 Complete Example: Running a Monitor

Let's walk through a real example step by step:

**Step 1: Edit locally**
```bash
# Make changes to your monitor script
vim methods/RL/monitoring/rl_environment_monitor_immediate.py
```

**Step 2: Sync to remote**
```bash
rsync -av methods/RL/monitoring/rl_environment_monitor_immediate.py \
    shurui.gui@dive7.engr.tamu.edu:/data/shurui.gui/Projects/Sys2Bench/methods/RL/monitoring/
```

**Step 3: Run in tmux session**
```bash
cd tmux_utils

# Navigate to project directory
./run_in_tmux.sh "cd /data/shurui.gui/Projects/Sys2Bench" monitor

# Activate conda environment (fish shell - no need to source)
./run_in_tmux.sh "conda activate sys2bench" monitor

# Run the script
./run_in_tmux.sh "python methods/RL/monitoring/rl_environment_monitor_immediate.py" monitor
```

**Step 4: Check the output**
```bash
# See what's happening
./check_tmux.sh monitor

# If needed, stop the process
./run_in_tmux.sh C-c monitor
```

## 🐛 Debugging TMUX Sessions

When things don't work as expected, debug step by step:

**1. Check if session exists:**
```bash
ssh user@server "tmux ls"
```

**2. Test with simple commands first:**
```bash
# Test echo
./run_in_tmux.sh "echo 'Hello World'" test

# Check result
./check_tmux.sh test
```

**3. Check the shell type:**
```bash
./run_in_tmux.sh "echo \$SHELL" test
./check_tmux.sh test
```

**4. Important: Use `tail -100` for tmux content**
- TMUX panes can be large with empty space at bottom
- Our scripts use appropriate viewing methods

## 🐟 Fish Shell Notes

Many servers use fish shell in tmux:
- `conda activate env` works directly (no sourcing needed)
- Different syntax than bash
- Our utilities handle this automatically

## 🎯 Quick Reference

```bash
# Edit locally
vim file.py

# Sync to remote
rsync -av file.py user@server:/path/

# Run in tmux
cd tmux_utils
./run_in_tmux.sh "command" session_name

# Check output
./check_tmux.sh session_name

# Stop process
./run_in_tmux.sh C-c session_name
```

## 💡 Tips

1. **Always edit locally** - Never modify files directly on remote
2. **Use meaningful session names** - Makes it easier to manage multiple tasks
3. **Check before running** - Always check session content before sending new commands
4. **Start simple** - Test with echo commands when debugging

---
*This workflow ensures consistency and prevents accidental file conflicts between local and remote.*