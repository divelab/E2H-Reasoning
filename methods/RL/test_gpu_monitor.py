#!/usr/bin/env python3
"""
Test GPU Monitor Script - Tests with 1 GPU requirement
"""

import subprocess
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import os
import sys

def check_gpu_availability(min_free_memory_gb=20, num_gpus_needed=1):
    """
    Check if enough GPUs are available with sufficient free memory
    
    Args:
        min_free_memory_gb: Minimum free memory in GB required per GPU
        num_gpus_needed: Number of GPUs needed
        
    Returns:
        tuple: (available, gpu_info) where available is bool and gpu_info is list of available GPU indices
    """
    try:
        # Run nvidia-smi to get GPU memory info
        cmd = "nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits"
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error running nvidia-smi: {result.stderr}")
            return False, []
        
        # Parse GPU info
        available_gpus = []
        print(f"GPU Memory Status:")
        for line in result.stdout.strip().split('\n'):
            gpu_id, free_memory_mb = line.split(', ')
            free_memory_gb = float(free_memory_mb) / 1024
            print(f"  GPU {gpu_id}: {free_memory_gb:.1f} GB free")
            
            if free_memory_gb >= min_free_memory_gb:
                available_gpus.append((int(gpu_id), free_memory_gb))
        
        print(f"\nGPUs with >= {min_free_memory_gb}GB free: {len(available_gpus)}")
        
        # Check if we have enough GPUs
        if len(available_gpus) >= num_gpus_needed:
            return True, available_gpus[:num_gpus_needed]
        else:
            return False, available_gpus
            
    except Exception as e:
        print(f"Error checking GPU availability: {e}")
        return False, []

def send_email_notification(recipient_email, available_gpus, server_name="dive7.engr.tamu.edu"):
    """
    Send email notification when GPUs are available
    """
    subject = f"[TEST] GPU Available on {server_name}"
    
    gpu_list = "\n".join([f"  - GPU {gpu_id}: {free_gb:.1f} GB free" for gpu_id, free_gb in available_gpus])
    
    body = f"""This is a TEST notification from the GPU monitor.

Available GPUs:
{gpu_list}

Server: {server_name}
Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

This test confirms that the monitoring script is working correctly.
When 2 GPUs become available, you'll receive a similar notification with the actual training command.
"""
    
    # Create a notification file
    notification_file = f"/data/shurui.gui/Projects/Sys2Bench/methods/RL/gpu_test_notification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(notification_file, 'w') as f:
        f.write(f"To: {recipient_email}\n")
        f.write(f"Subject: {subject}\n\n")
        f.write(body)
    
    print(f"\nNotification saved to: {notification_file}")
    
    # Try to send email using mail command
    try:
        # First, let's check if mail command exists
        mail_check = subprocess.run(['which', 'mail'], capture_output=True, text=True)
        if mail_check.returncode == 0:
            print(f"Mail command found at: {mail_check.stdout.strip()}")
            
            # Try sending email
            mail_cmd = f'echo "{body}" | mail -s "{subject}" {recipient_email}'
            result = subprocess.run(mail_cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✓ Email sent successfully to {recipient_email}")
            else:
                print(f"✗ Failed to send email. Return code: {result.returncode}")
                print(f"  Error: {result.stderr}")
        else:
            print("✗ Mail command not found on system")
            
            # Try using sendmail as alternative
            sendmail_check = subprocess.run(['which', 'sendmail'], capture_output=True, text=True)
            if sendmail_check.returncode == 0:
                print(f"Sendmail found at: {sendmail_check.stdout.strip()}")
                # Could implement sendmail here if needed
            else:
                print("✗ Sendmail not found either")
    except Exception as e:
        print(f"✗ Error sending email: {e}")
    
    # Also create a more visible notification
    print("\n" + "="*60)
    print("GPU NOTIFICATION - TEST")
    print("="*60)
    print(f"GPUs are available! {len(available_gpus)} GPU(s) found with sufficient memory")
    for gpu_id, free_gb in available_gpus:
        print(f"  GPU {gpu_id}: {free_gb:.1f} GB free")
    print("="*60)
    
    return notification_file

# Run the test
if __name__ == "__main__":
    print("GPU Monitor Test - Checking for 1 GPU with 20GB+ free memory")
    print("=" * 60)
    
    RECIPIENT_EMAIL = "citrinegui@gmail.com"
    
    # Check for GPUs
    available, gpu_info = check_gpu_availability(min_free_memory_gb=20, num_gpus_needed=1)
    
    if available:
        print(f"\n✓ Found suitable GPU(s)!")
        notification_file = send_email_notification(RECIPIENT_EMAIL, gpu_info)
        print(f"\nTest complete. Check notification file: {notification_file}")
    else:
        print(f"\n✗ No suitable GPUs found (need 1 GPU with 20GB+ free)")
        if gpu_info:
            print("  GPUs with sufficient memory:")
            for gpu_id, free_gb in gpu_info:
                print(f"    - GPU {gpu_id}: {free_gb:.1f} GB free")