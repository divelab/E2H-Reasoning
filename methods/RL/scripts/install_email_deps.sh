#!/bin/bash

# Install email dependencies in sys2bench conda environment

echo "Installing email dependencies for GPU monitor..."

# Activate conda environment
source /data/shurui.gui/mambaforge/etc/profile.d/conda.sh
conda activate sys2bench

# Install required Python packages
pip install cryptography

echo "Dependencies installed successfully!"
echo ""
echo "Next steps:"
echo "1. Run: python setup_email.py"
echo "2. Configure your email credentials"
echo "3. Run: python gpu_monitor_email.py"