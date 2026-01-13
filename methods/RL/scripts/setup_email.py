#!/usr/bin/env python3
"""
Setup script for email configuration
Creates a secure credentials file for email notifications
"""

import os
import json
import getpass
from cryptography.fernet import Fernet

def setup_email_credentials():
    """Setup email credentials securely"""
    print("Email Configuration Setup")
    print("=" * 50)
    print("\nThis script will help you set up email notifications for GPU monitoring.")
    print("Your credentials will be encrypted and stored locally.\n")
    
    # Get email configuration
    print("Please provide the following information:")
    
    smtp_server = input("SMTP server (e.g., smtp.gmail.com): ").strip()
    smtp_port = input("SMTP port (e.g., 587 for TLS, 465 for SSL): ").strip()
    
    sender_email = input("Sender email address: ").strip()
    sender_password = getpass.getpass("Sender email password/app password: ").replace(' ', '')
    
    recipient_email = input("Recipient email address (default: citrinegui@gmail.com): ").strip()
    if not recipient_email:
        recipient_email = "citrinegui@gmail.com"
    
    use_tls = input("Use TLS? (y/n, default: y): ").strip().lower()
    use_tls = use_tls != 'n'
    
    # Generate encryption key
    key = Fernet.generate_key()
    cipher_suite = Fernet(key)
    
    # Encrypt password
    encrypted_password = cipher_suite.encrypt(sender_password.encode())
    
    # Create config
    config = {
        "smtp_server": smtp_server,
        "smtp_port": int(smtp_port),
        "sender_email": sender_email,
        "encrypted_password": encrypted_password.decode(),
        "recipient_email": recipient_email,
        "use_tls": use_tls
    }
    
    # Save configuration
    config_dir = os.path.expanduser("~/.sys2bench")
    os.makedirs(config_dir, exist_ok=True)
    
    # Save key separately
    key_file = os.path.join(config_dir, "email.key")
    with open(key_file, 'wb') as f:
        f.write(key)
    os.chmod(key_file, 0o600)  # Read/write for owner only
    
    # Save config
    config_file = os.path.join(config_dir, "email_config.json")
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    os.chmod(config_file, 0o600)  # Read/write for owner only
    
    print(f"\n✓ Configuration saved to {config_dir}")
    print(f"  - Encryption key: {key_file}")
    print(f"  - Email config: {config_file}")
    
    print("\n" + "="*50)
    print("Setup complete!")
    print("\nFor Gmail users:")
    print("  - You may need to use an App Password instead of your regular password")
    print("  - Enable 2FA and generate an app password at: https://myaccount.google.com/apppasswords")
    print("\nFor other email providers:")
    print("  - Check their documentation for SMTP settings and app-specific passwords")
    
    return config_dir

if __name__ == "__main__":
    setup_email_credentials()