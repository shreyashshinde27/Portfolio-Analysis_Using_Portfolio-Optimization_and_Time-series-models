#!/usr/bin/env python3
"""
StockStream Pro - Launch Script
This script provides an easy way to launch the StockStream Pro application
with proper environment setup and error handling.
"""

import sys
import subprocess
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def check_virtual_environment():
    """Check if running in virtual environment"""
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Running in virtual environment")
        return True
    else:
        print("⚠️  Warning: Not running in virtual environment")
        print("   Consider creating one: python -m venv .venv")
        return False

def install_requirements():
    """Install required packages"""
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print("❌ Error: requirements.txt not found")
        return False
    
    print("📦 Installing requirements...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True, text=True)
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        print("Try running: pip install -r requirements.txt")
        return False

def check_data_file():
    """Check if ticker data file exists"""
    data_file = Path("StockStreamTickersData.csv")
    if not data_file.exists():
        print("❌ Error: StockStreamTickersData.csv not found")
        print("   Please ensure the data file is in the project directory")
        return False
    print("✅ Data file found")
    return True

def launch_app():
    """Launch the Streamlit application"""
    print("🚀 Launching Stock Analyzer...")
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching application: {e}")
        return False
    return True

def main():
    """Main function"""
    print("=" * 50)
    print("📈 Stock Analyzer - Launch Script")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check virtual environment
    check_virtual_environment()
    
    # Check data file
    if not check_data_file():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("⚠️  Continuing without installing requirements...")
        print("   Make sure all packages are installed manually")
    
    print("\n" + "=" * 50)
    print("🎯 All checks passed! Starting application...")
    print("=" * 50)
    
    # Launch application
    launch_app()

if __name__ == "__main__":
    main()