#!/usr/bin/env python3
"""
Setup script for CSV-QA-ChatBot Authentication System
"""

import subprocess
import sys
import os
from pathlib import Path

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}")
        return False

def create_directories():
    """Create necessary directories for the authentication system"""
    directories = [
        "user_data",
        "user_data/saved_chats"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def main():
    """Main setup function"""
    print("🚀 Setting up CSV-QA-ChatBot Authentication System...")
    print("=" * 50)
    
    # Create directories
    print("\n📁 Creating directories...")
    create_directories()
    
    # Install required packages
    print("\n📦 Installing required packages...")
    packages = [
        "scikit-learn>=1.3.0"
    ]
    
    success_count = 0
    for package in packages:
        if install_package(package):
            success_count += 1
    
    print(f"\n📊 Installation Summary:")
    print(f"✅ Successfully installed: {success_count}/{len(packages)} packages")
    print(f"✅ Directories created: {len(['user_data', 'user_data/saved_chats'])}")
    
    if success_count == len(packages):
        print("\n🎉 Authentication system setup completed successfully!")
        print("\n🔐 Features available:")
        print("   • Guest user authentication")
        print("   • Chat session saving")
        print("   • Chat history management")
        print("   • SQLite database storage")
        print("\n💡 Next steps:")
        print("   1. Run 'streamlit run main.py'")
        print("   2. Go to the '🔐 Account' tab")
        print("   3. Sign in as a guest user")
        print("   4. Start chatting and save your sessions!")
    else:
        print("\n⚠️  Some packages failed to install. Please check the errors above.")
        print("   You can still use the basic features without authentication.")

if __name__ == "__main__":
    main()
