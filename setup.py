#!/usr/bin/env python3
"""
Setup script for the Human Classifier project.
This script installs dependencies and sets up the project structure.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def install_python_dependencies():
    """Install Python dependencies."""
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found")
        return False
    
    return run_command(f"{sys.executable} -m pip install -r requirements.txt", "Installing Python dependencies")


def setup_frontend():
    """Setup the React frontend."""
    frontend_dir = Path("src/frontend")
    
    if not frontend_dir.exists():
        print("❌ Frontend directory not found")
        return False
    
    # Check if Node.js is installed
    if not run_command("node --version", "Checking Node.js installation"):
        print("❌ Node.js is not installed. Please install Node.js from https://nodejs.org/")
        return False
    
    # Check if npm is installed
    if not run_command("npm --version", "Checking npm installation"):
        print("❌ npm is not installed. Please install npm")
        return False
    
    # Install frontend dependencies
    return run_command("npm install", "Installing frontend dependencies")


def create_directories():
    """Create necessary directories."""
    directories = [
        "models",
        "logs",
        "src/model",
        "src/data", 
        "src/api",
        "src/frontend/src/components",
        "src/frontend/src/services"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Project directories created")
    return True


def main():
    """Main setup function."""
    print("🎯 Human Classifier Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Create directories
    if not create_directories():
        return False
    
    # Install Python dependencies
    if not install_python_dependencies():
        return False
    
    # Setup frontend
    if not setup_frontend():
        print("⚠️  Frontend setup failed, but you can continue with backend-only usage")
    
    print("\n✨ Setup completed successfully!")
    print("\n📝 Next Steps:")
    print("1. Add video data to dataset/ directories")
    print("2. Train the model: python3 src/model/train.py")
    print("3. Start the demo: python3 run_demo.py")
    print("\n📚 For detailed instructions, see README.md")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
