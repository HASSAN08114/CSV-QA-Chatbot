#!/usr/bin/env python3
"""
Quick Setup Script for Fine-tuning
Run this to install required packages and set up the environment
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Installed {package}")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}")
        return False

def main():
    print("🚀 Setting up Fine-tuning Environment for CSV Q&A ChatBot")
    print("=" * 60)
    
    # Required packages for fine-tuning
    packages = [
        "peft>=0.5.0",
        "bitsandbytes>=0.41.0", 
        "accelerate>=0.20.0",
        "datasets>=2.12.0",
        "scikit-learn>=1.3.0",
        "wandb>=0.15.0"
    ]
    
    print("📦 Installing required packages...")
    success_count = 0
    
    for package in packages:
        if install_package(package):
            success_count += 1
    
    print(f"\n📊 Installation Summary: {success_count}/{len(packages)} packages installed")
    
    if success_count == len(packages):
        print("✅ All packages installed successfully!")
        
        # Create training directory
        os.makedirs("training_data", exist_ok=True)
        print("📁 Created training_data directory")
        
        print("\n🎉 Setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Run the main app: streamlit run main.py")
        print("2. Use the chat interface to collect training data")
        print("3. Run the training interface: streamlit run training_interface.py")
        print("4. Train your fine-tuned model!")
        
    else:
        print("⚠️ Some packages failed to install. Please check your internet connection and try again.")
        print("You can manually install the missing packages using:")
        print("pip install peft bitsandbytes accelerate datasets scikit-learn wandb")

if __name__ == "__main__":
    main()
