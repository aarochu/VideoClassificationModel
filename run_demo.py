#!/usr/bin/env python3
"""
Demo script to run the Human Classifier application.
This script starts both the backend API and provides instructions for the frontend.
"""

import subprocess
import sys
import time
import webbrowser
from pathlib import Path
import os


def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import torch
        import fastapi
        import uvicorn
        print("✅ Python dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please run: pip install -r requirements.txt")
        return False


def check_model():
    """Check if a trained model exists."""
    models_dir = Path("models")
    if models_dir.exists():
        model_files = list(models_dir.glob("*.pth"))
        if model_files:
            print(f"✅ Found trained model: {model_files[0]}")
            return True
    
    print("❌ No trained model found")
    print("Please train a model first: python src/model/train.py")
    return False


def start_backend():
    """Start the FastAPI backend server."""
    print("🚀 Starting backend API server...")
    
    # Change to the API directory
    api_dir = Path("src/api")
    if not api_dir.exists():
        print("❌ API directory not found")
        return None
    
    try:
        # Start the server
        process = subprocess.Popen(
            [sys.executable, "main.py"],
            cwd=api_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait a moment for the server to start
        time.sleep(3)
        
        if process.poll() is None:
            print("✅ Backend server started on http://localhost:8000")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Failed to start backend: {stderr.decode()}")
            return None
            
    except Exception as e:
        print(f"❌ Error starting backend: {e}")
        return None


def start_frontend():
    """Start the React frontend."""
    frontend_dir = Path("src/frontend")
    
    if not frontend_dir.exists():
        print("❌ Frontend directory not found")
        return None
    
    # Check if node_modules exists
    if not (frontend_dir / "node_modules").exists():
        print("📦 Installing frontend dependencies...")
        try:
            subprocess.run(["npm", "install"], cwd=frontend_dir, check=True)
            print("✅ Frontend dependencies installed")
        except subprocess.CalledProcessError:
            print("❌ Failed to install frontend dependencies")
            return None
    
    print("🚀 Starting frontend development server...")
    try:
        process = subprocess.Popen(
            ["npm", "start"],
            cwd=frontend_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for the server to start
        time.sleep(5)
        
        if process.poll() is None:
            print("✅ Frontend server started on http://localhost:3000")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Failed to start frontend: {stderr.decode()}")
            return None
            
    except Exception as e:
        print(f"❌ Error starting frontend: {e}")
        return None


def main():
    """Main demo function."""
    print("🎯 Human Classifier Demo")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Check if model exists
    if not check_model():
        print("\n📝 To train a model:")
        print("1. Add video data to the dataset/ directory")
        print("2. Run: python src/model/train.py")
        print("3. Then run this demo script again")
        return
    
    print("\n🚀 Starting the Human Classifier application...")
    
    # Start backend
    backend_process = start_backend()
    if not backend_process:
        return
    
    # Start frontend
    frontend_process = start_frontend()
    if not frontend_process:
        backend_process.terminate()
        return
    
    print("\n✨ Application started successfully!")
    print("\n📱 Access the application:")
    print("   Frontend: http://localhost:3000")
    print("   Backend API: http://localhost:8000")
    print("   API Docs: http://localhost:8000/docs")
    
    # Try to open the browser
    try:
        time.sleep(2)
        webbrowser.open("http://localhost:3000")
        print("🌐 Opened browser to http://localhost:3000")
    except:
        print("🌐 Please open http://localhost:3000 in your browser")
    
    print("\n⏹️  Press Ctrl+C to stop the servers")
    
    try:
        # Wait for user to stop
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping servers...")
        
        if backend_process:
            backend_process.terminate()
            print("✅ Backend server stopped")
        
        if frontend_process:
            frontend_process.terminate()
            print("✅ Frontend server stopped")
        
        print("👋 Demo ended. Thank you!")


if __name__ == "__main__":
    main()
