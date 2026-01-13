#!/usr/bin/env python3
"""
RL Environment Monitor with Immediate GPU Preparation
Monitors GPU availability and immediately prepares any GPU with sufficient memory
"""

import subprocess
import time
import smtplib
import ssl
import json
import os
import signal
import sys
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from cryptography.fernet import Fernet

# Global variables for tracking
prep_processes = {}  # Dict mapping GPU ID to process
prepared_gpus = set()  # Set of GPU IDs that are already prepared

def signal_handler(sig, frame):
    """Handle cleanup on exit"""
    print(f"\n[{datetime.now()}] Received shutdown signal. Cleaning up...")
    cleanup_prep_processes()
    sys.exit(0)

def cleanup_prep_processes():
    """Kill all preparation processes"""
    for gpu_id, proc in prep_processes.items():
        if proc.poll() is None:  # Process is still running
            print(f"Terminating preparation process for GPU {gpu_id} (PID: {proc.pid})")
            proc.terminate()
            proc.wait()
    prep_processes.clear()
    prepared_gpus.clear()

def load_email_config():
    """Load email configuration from encrypted storage"""
    config_dir = os.path.expanduser("~/.sys2bench")
    config_file = os.path.join(config_dir, "email_config.json")
    key_file = os.path.join(config_dir, "email.key")
    
    if not os.path.exists(config_file) or not os.path.exists(key_file):
        raise FileNotFoundError(
            "Email configuration not found. Please run setup_email.py first."
        )
    
    # Load encryption key
    with open(key_file, 'rb') as f:
        key = f.read()
    cipher_suite = Fernet(key)
    
    # Load config
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Decrypt password
    config['password'] = cipher_suite.decrypt(
        config['encrypted_password'].encode()
    ).decode()
    
    return config

def check_all_gpus(min_free_memory_gb=50):
    """Check all GPUs and return those with sufficient free memory"""
    try:
        cmd = "nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits"
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"Error running nvidia-smi: {result.stderr}")
            return []
        
        # Parse GPU info
        available_gpus = []
        for line in result.stdout.strip().split('\n'):
            gpu_id, free_memory_mb = line.split(', ')
            gpu_id = int(gpu_id)
            free_memory_gb = float(free_memory_mb) / 1024
            
            # Only consider GPUs that aren't already prepared and have enough memory
            if gpu_id not in prepared_gpus and free_memory_gb >= min_free_memory_gb:
                available_gpus.append((gpu_id, free_memory_gb))
        
        return available_gpus
            
    except Exception as e:
        print(f"Error checking GPU availability: {e}")
        return []

def prepare_gpu(gpu_id, free_gb, memory_to_prepare=45):
    """Start preparation process for a single GPU"""
    # Script is now at methods/RL/sys_rl.py
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prep_script = os.path.join(script_dir, "sys_rl.py")
    
    # Make sure the preparation script exists
    if not os.path.exists(prep_script):
        print(f"Error: RL preparation script not found at {prep_script}")
        return False
    
    print(f"[{datetime.now()}] Starting RL environment preparation on GPU {gpu_id} ({free_gb:.1f}GB free)...")
    
    # Python command that will work on the remote system
    python_cmd = "python3"
    
    # Use 45GB to leave some buffer (for 50GB threshold)
    # New parameters: --sc for system capacity, --data for dataset
    cmd = [python_cmd, prep_script, str(gpu_id), "--sc", str(memory_to_prepare), "--data", "countdown"]
    
    try:
        # Start the process in the background
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid  # Create new process group for easier cleanup
        )
        
        # Give it a moment to allocate memory
        time.sleep(3)
        
        # Check if process is still running
        if proc.poll() is not None:
            stdout, stderr = proc.communicate()
            print(f"✗ Environment preparation failed for GPU {gpu_id}")
            print(f"  stdout: {stdout.decode()}")
            print(f"  stderr: {stderr.decode()}")
            return False
        
        # Success - track the process
        prep_processes[gpu_id] = proc
        prepared_gpus.add(gpu_id)
        print(f"✓ RL environment preparation started for GPU {gpu_id} (PID: {proc.pid})")
        
        # Save PIDs
        save_preparation_pids()
        
        return True
            
    except Exception as e:
        print(f"Error starting RL preparation for GPU {gpu_id}: {e}")
        return False

def save_preparation_pids():
    """Save preparation PIDs to file for later cleanup"""
    pid_file = "rl_prep_pids.txt"
    with open(pid_file, 'w') as f:
        for gpu_id, proc in prep_processes.items():
            f.write(f"{proc.pid} # GPU {gpu_id}\n")
    print(f"Preparation PIDs saved to {pid_file}")

def send_email_notification(config, prepared_gpu_list, server_name="dive7.engr.tamu.edu"):
    """Send email notification about prepared GPUs"""
    
    gpu_list = "\n".join([f"  - GPU {gpu_id}: {free_gb:.1f}GB free (now prepared)" 
                         for gpu_id, free_gb in prepared_gpu_list])
    
    subject = f"✅ Both GPUs Ready on {server_name} - Training Can Begin!"
    
    # Get list of prepared GPU IDs
    gpu_ids = [gpu_id for gpu_id, _ in prepared_gpu_list]
    cuda_devices = ','.join(map(str, gpu_ids))
    
    # Adjust command based on number of GPUs
    if len(gpu_ids) >= 2:
        training_note = "You have 2+ GPUs ready for full training!"
    else:
        training_note = f"You have {len(gpu_ids)} GPU(s) ready. Waiting for more GPUs for full training."
    
    command = f"""WANDB_PROJECT=Sys2Bench-VarReg ROOT_PATH=/data/shurui.gui/Projects/Sys2Bench CUDA_VISIBLE_DEVICES={cuda_devices} accelerate launch \\
    --num_processes 1 \\
    --main_process_port=29850 \\
    --config_file methods/RL/deep_speed.yaml \\
    methods/RL/main.py \\
    mode=train \\
    task=countdown6 \\
    algorithm=grpo \\
    algorithm.training.curriculum_schedule=variance_regularized \\
    model=qwen15 \\
    algorithm.training.per_device_train_batch_size=2 \\
    algorithm.training.max_steps=1600"""
    
    body = f"""🎉 Great news! Both GPUs are now secured and ready for your variance regularized training on {server_name}.

Secured GPUs:
{gpu_list}

✅ Both GPUs are ready for full GRPO training with VLLM!

⚠️ IMPORTANT: The GPUs are being held by preparation processes.

To use the GPUs:
1. SSH to {server_name}
2. Kill the preparation processes:
   cat rl_prep_pids.txt | xargs kill
3. cd /data/shurui.gui/Projects/Sys2Bench
4. source /data/shurui.gui/mambaforge/etc/profile.d/conda.sh
5. conda activate sys2bench
6. Run your training command:

{command}

Monitor Status:
- Total GPUs prepared: {len(prepared_gpus)}
- Active preparation processes: {len(prep_processes)}
- Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

The monitor continues to run and will prepare additional GPUs as they become available.
"""
    
    # Create message
    message = MIMEMultipart()
    message["From"] = config['sender_email']
    message["To"] = config['recipient_email']
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))
    
    # Send email
    try:
        if config['use_tls']:
            # TLS connection
            context = ssl.create_default_context()
            with smtplib.SMTP(config['smtp_server'], config['smtp_port']) as server:
                server.starttls(context=context)
                server.login(config['sender_email'], config['password'])
                server.send_message(message)
        else:
            # SSL connection
            context = ssl.create_default_context()
            with smtplib.SMTP_SSL(config['smtp_server'], config['smtp_port'], 
                                   context=context) as server:
                server.login(config['sender_email'], config['password'])
                server.send_message(message)
        
        print(f"✓ Email sent successfully to {config['recipient_email']}")
        return True
        
    except Exception as e:
        print(f"✗ Failed to send email: {e}")
        return False

def monitor_and_prepare_gpus_immediate(check_interval_seconds=60, min_free_memory_gb=50, 
                                     memory_to_prepare=45, max_gpus=2):
    """Monitor GPUs and immediately prepare any with sufficient memory (up to max_gpus)"""
    
    # Set up signal handlers for cleanup
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Load email configuration
    try:
        email_config = load_email_config()
        print(f"✓ Email configuration loaded")
        print(f"  Will send notifications to: {email_config['recipient_email']}")
    except Exception as e:
        print(f"✗ Error loading email configuration: {e}")
        print("  Please run setup_email.py first to configure email settings")
        return
    
    print(f"\n=== RL Environment Monitor (Immediate Mode) ===")
    print(f"Will immediately prepare any GPU with {min_free_memory_gb}GB+ free memory")
    print(f"Maximum GPUs to prepare: {max_gpus}")
    print(f"Allocation size: {memory_to_prepare}GB per GPU")
    print(f"Check interval: {check_interval_seconds} seconds")
    print("-" * 60)
    
    check_count = 0
    last_email_count = 0  # Track when to send email updates
    
    while True:
        check_count += 1
        print(f"\n[Check #{check_count}] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Check for dead processes and clean them up
        dead_gpus = []
        for gpu_id, proc in prep_processes.items():
            if proc.poll() is not None:
                print(f"⚠️  Preparation process for GPU {gpu_id} has died")
                dead_gpus.append(gpu_id)
        
        for gpu_id in dead_gpus:
            del prep_processes[gpu_id]
            prepared_gpus.remove(gpu_id)
        
        # Check if we've reached our limit
        if len(prepared_gpus) >= max_gpus:
            print(f"Already holding {max_gpus} GPUs - target reached!")
            # Just monitor that they're still alive
            if check_count % 10 == 0:  # Status update every 10 checks
                print(f"\nMaintaining {len(prepared_gpus)} GPUs: {sorted(prepared_gpus)}")
                subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.free", 
                              "--format=csv,noheader"])
        else:
            # Check all GPUs for availability
            available_gpus = check_all_gpus(min_free_memory_gb)
            
            if available_gpus:
                print(f"Found {len(available_gpus)} GPU(s) with {min_free_memory_gb}GB+ free")
                
                # Prepare GPUs up to our limit
                newly_prepared = []
                for gpu_id, free_gb in available_gpus:
                    if len(prepared_gpus) >= max_gpus:
                        print(f"Reached target of {max_gpus} GPUs, stopping preparation")
                        break
                    if prepare_gpu(gpu_id, free_gb, memory_to_prepare):
                        newly_prepared.append((gpu_id, free_gb))
            
                # Check if we've reached our target
                if len(prepared_gpus) >= max_gpus:
                    print(f"\n🎯 SUCCESS: Secured {max_gpus} GPUs! Monitor will continue running to maintain them.")
                    
                    # Send email only when we have all GPUs ready
                    if len(prepared_gpus) != last_email_count:
                        # Get list of all prepared GPUs (we'll show them as having ~50GB originally)
                        all_prepared_list = [(gpu_id, 50.0) for gpu_id in sorted(prepared_gpus)]
                        send_email_notification(email_config, all_prepared_list)
                        last_email_count = len(prepared_gpus)
                        print("✉️  Email notification sent - both GPUs are ready!")
        
        # Status update
        if prepared_gpus:
            print(f"Currently holding {len(prepared_gpus)} GPU(s): {sorted(prepared_gpus)}")
        else:
            print("No GPUs currently prepared")
        
        # Show current GPU status
        if check_count % 5 == 0:  # Every 5 checks
            print("\nCurrent GPU status:")
            subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.free", 
                          "--format=csv,noheader"])
        
        print(f"Next check in {check_interval_seconds} seconds...")
        time.sleep(check_interval_seconds)

if __name__ == "__main__":
    import sys
    
    # Configuration
    CHECK_INTERVAL = 60  # Check every minute
    MIN_FREE_MEMORY_GB = 50  # Need at least 50GB free
    MEMORY_TO_PREPARE = 45  # Prepare with 45GB (leaving 5GB buffer)
    
    # Allow command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            print("TEST MODE: Using lower thresholds")
            MIN_FREE_MEMORY_GB = 20
            MEMORY_TO_PREPARE = 15
            CHECK_INTERVAL = 30
        elif sys.argv[1] == "--help":
            print("Usage: python rl_environment_monitor_immediate.py [--test]")
            print("  --test: Use lower memory thresholds for testing")
            sys.exit(0)
    
    print("RL Environment Monitor - Immediate GPU Preparation")
    print("=" * 60)
    
    try:
        monitor_and_prepare_gpus_immediate(
            check_interval_seconds=CHECK_INTERVAL,
            min_free_memory_gb=MIN_FREE_MEMORY_GB,
            memory_to_prepare=MEMORY_TO_PREPARE,
            max_gpus=2  # Only secure 2 GPUs
        )
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")
        cleanup_prep_processes()
    except Exception as e:
        print(f"\nError during monitoring: {e}")
        cleanup_prep_processes()