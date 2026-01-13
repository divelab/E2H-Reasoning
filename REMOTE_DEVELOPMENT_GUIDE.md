# Remote Development Guide for Sys2Bench

This guide consolidates all remote development workflows and best practices for working with Sys2Bench on remote servers. It is extracted from various documentation files to provide a single reference.

## Table of Contents

1. [🚀 Overview](#-overview)
2. [🔄 The Basic Workflow](#-the-basic-workflow)
3. [📝 Step-by-Step Development Process](#-step-by-step-development-process)
4. [🖥️ TMUX Session Management](#️-tmux-session-management)
5. [🔧 GPU Management for RL Tasks](#-gpu-management-for-rl-tasks)
6. [🐟 Fish Shell Considerations](#-fish-shell-considerations)
7. [📋 Common Workflows](#-common-workflows)
8. [🛠️ Troubleshooting](#️-troubleshooting)
9. [🎯 Quick Reference Commands](#-quick-reference-commands)

## 🚀 Overview

Most Sys2Bench experiments require GPU access and long-running processes. This guide covers the essential workflow for developing locally and running experiments on remote servers.

**Key Principle**: Always edit files locally, sync to remote, and run on remote servers.

## 🔄 The Basic Workflow

1. **Edit locally** → 2. **Sync to remote** → 3. **Run on remote** → 4. **Check results**

This workflow ensures:
- Version control consistency
- No accidental file conflicts
- Clean development environment
- Ability to work offline

## 📝 Step-by-Step Development Process

### Step 1: Local Development

Always edit files on your local machine:
```bash
# Edit your code locally
vim methods/RL/my_script.py

# Test syntax locally
python -m py_compile methods/RL/my_script.py

# Commit changes
git add methods/RL/my_script.py
git commit -m "Add feature X"
```

### Step 2: Sync to Remote

Use rsync to upload files:
```bash
# Single file
rsync -av my_script.py user@server:/path/to/destination/

# Multiple files
rsync -av file1.py file2.py user@server:/path/to/destination/

# Entire directory
rsync -av methods/RL/ user@server:/path/to/project/methods/RL/

# Example for Sys2Bench
rsync -av methods/RL/sys_rl.py shurui.gui@dive7.engr.tamu.edu:/data/shurui.gui/Projects/Sys2Bench/methods/RL/
```

### Step 3: Run on Remote

#### Option A: Direct SSH (for quick commands)
```bash
ssh user@server "cd /path/to/project && python my_script.py"
```

#### Option B: Using TMUX (recommended for long-running processes)
```bash
# First sync tmux utilities
rsync -av tmux_utils/ user@server:/path/to/project/tmux_utils/

# Then use them
cd tmux_utils
./run_in_tmux.sh "python my_script.py" session_name
./check_tmux.sh session_name
```

## 🖥️ TMUX Session Management

### Available Scripts

The `tmux_utils/` directory contains helper scripts:

- **`run_in_tmux.sh`** - Execute commands in tmux sessions
- **`check_tmux.sh`** - View tmux session content

### Basic Usage

```bash
# Create session and run command
./run_in_tmux.sh "python my_script.py" session_name

# Check output
./check_tmux.sh session_name

# Send control keys
./run_in_tmux.sh C-c session_name  # Send Ctrl+C
./run_in_tmux.sh C-z session_name  # Send Ctrl+Z
```

### Common TMUX Workflows

1. **Start a Long-Running Process**
```bash
# Navigate to project
./run_in_tmux.sh "cd /data/shurui.gui/Projects/Sys2Bench" monitor

# Activate environment
./run_in_tmux.sh "conda activate sys2bench" monitor

# Run script
./run_in_tmux.sh "python methods/RL/monitoring/rl_environment_monitor_immediate.py" monitor

# Check output
./check_tmux.sh monitor
```

2. **Multiple Commands in Sequence**
```bash
./run_in_tmux.sh "cd /path/to/project" session
./run_in_tmux.sh "conda activate sys2bench" session
./run_in_tmux.sh "python script.py" session
```

## 🔧 GPU Management for RL Tasks

### GPU Requirements

- **GRPO Training**: Requires 2 GPUs (1 for model, 1 for VLLM ~20GB each)
- **Other methods**: Usually 1 GPU is sufficient

### GPU Monitoring Workflow

1. **Check GPU Availability**
```bash
# SSH to server first
ssh shurui.gui@dive7.engr.tamu.edu

# Check GPUs
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader
```

2. **Use GPU Monitor for Automatic Reservation**
```bash
# Using tmux utilities
cd tmux_utils
./run_in_tmux.sh "cd /data/shurui.gui/Projects/Sys2Bench" gpu_monitor
./run_in_tmux.sh "conda activate sys2bench" gpu_monitor
./run_in_tmux.sh "python methods/RL/monitoring/rl_environment_monitor_immediate.py" gpu_monitor
```

3. **Start Training When GPUs Ready**
```bash
# Kill occupation processes
cat rl_prep_pids.txt | xargs kill

# Run training
WANDB_PROJECT=Sys2Bench ROOT_PATH=/data/shurui.gui/Projects/Sys2Bench \
CUDA_VISIBLE_DEVICES=0,1 accelerate launch \
    --num_processes 1 \
    --config_file methods/RL/deep_speed.yaml \
    methods/RL/main.py \
    mode=train \
    task=countdown6 \
    algorithm=grpo \
    model=qwen15
```

## 🐟 Fish Shell Considerations

Many servers use fish shell in tmux sessions:

- **No sourcing needed**: `conda activate env` works directly
- **Different syntax**: Be aware of fish vs bash differences
- **Environment activation**: Simply use `conda activate sys2bench`

## 📋 Common Workflows

### Workflow 1: Running Standard Experiments

```bash
# 1. Edit locally
vim methods/CoT/gsm8k/inference.py

# 2. Sync to remote
rsync -av methods/CoT/gsm8k/inference.py user@server:/path/to/Sys2Bench/methods/CoT/gsm8k/

# 3. Run in tmux
cd tmux_utils
./run_in_tmux.sh "cd /path/to/Sys2Bench" exp1
./run_in_tmux.sh "conda activate sys2bench" exp1
./run_in_tmux.sh "bash methods/CoT/gsm8k/cot.sh" exp1

# 4. Monitor
./check_tmux.sh exp1
```

### Workflow 2: RL Training with GPU Monitoring

```bash
# 1. Start GPU monitor
cd tmux_utils
./run_in_tmux.sh "python methods/RL/monitoring/rl_environment_monitor_immediate.py" gpu_mon

# 2. Wait for notification (email or check manually)
./check_tmux.sh gpu_mon

# 3. Kill occupation and start training
ssh user@server
cat rl_prep_pids.txt | xargs kill
# Run accelerate launch command...
```

### Workflow 3: Debugging Failed Runs

```bash
# 1. Check session status
./check_tmux.sh failed_session

# 2. Test with simple commands
./run_in_tmux.sh "echo 'Test'" test
./check_tmux.sh test

# 3. Check environment
./run_in_tmux.sh "which python" test
./run_in_tmux.sh "conda info --envs" test
```

## 🛠️ Troubleshooting

### Common Issues and Solutions

1. **VPN Connection Required**
   - Error: `Could not resolve hostname`
   - Solution: Connect to university VPN first

2. **Session Already Exists**
   - Error: `duplicate session: session_name`
   - Solution: Use different name or `tmux kill-session -t session_name`

3. **Can't See Output**
   - Issue: Empty or incomplete output
   - Solution: Wait a moment, tmux needs time to update

4. **Command Not Found**
   - Issue: Python/conda not found
   - Solution: Activate environment first

5. **GPU Memory Issues**
   - Check with: `nvidia-smi`
   - Solution: Wait for GPUs to free up or use monitor

### Debug Commands

```bash
# List all tmux sessions
ssh user@server "tmux ls"

# Check shell type
./run_in_tmux.sh "echo \$SHELL" test

# Check conda environments
./run_in_tmux.sh "conda info --envs" test

# Kill stuck session
ssh user@server "tmux kill-session -t session_name"
```

## 🎯 Quick Reference Commands

### Essential Commands
```bash
# Edit locally
vim file.py

# Sync to remote
rsync -av file.py user@server:/path/

# Run in tmux
./run_in_tmux.sh "command" session

# Check output
./check_tmux.sh session

# Stop process
./run_in_tmux.sh C-c session
```

### GPU Commands
```bash
# Check GPUs
nvidia-smi

# Query specific info
nvidia-smi --query-gpu=index,memory.free --format=csv

# Monitor continuously
watch -n 1 nvidia-smi
```

### Environment Commands
```bash
# Activate Sys2Bench
conda activate sys2bench

# Set environment variables
export ROOT_PATH=/path/to/Sys2Bench
export OPENAI_API_KEY="your-key"
```

## 💡 Best Practices

1. **Always edit locally** - Never modify files directly on remote
2. **Use meaningful session names** - e.g., `rl_train_countdown`, `exp_gsm8k`
3. **Check before running** - Always check session content first
4. **Document your runs** - Keep notes on what's running where
5. **Clean up sessions** - Kill unused tmux sessions
6. **Use version control** - Commit changes before major experiments

---
*This guide consolidates all remote development information from various documentation files including CLAUDE.md, CLAUDE_RL.md, and tmux_utils/README.md for easy reference.*