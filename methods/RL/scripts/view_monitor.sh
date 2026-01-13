#!/bin/bash
# Quick script to view the GPU monitor tmux session

echo "Connecting to GPU monitor tmux session..."
echo "Press Ctrl+B then D to detach from tmux"
echo ""

ssh -t shurui.gui@dive7.engr.tamu.edu 'tmux attach -t gpu_monitor'