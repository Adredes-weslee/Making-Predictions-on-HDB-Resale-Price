"""
HDB Resale Price Prediction Application
======================================
This is the entry point for the Streamlit application that provides
tools to explore HDB resale data and make price predictions.

This file serves as a wrapper around the full implementation in app/main.py.
It ensures proper path configuration and simplifies deployment.

Usage:
    streamlit run app.py
"""
import os
import sys
from pathlib import Path

# Ensure the application runs consistently regardless of the current working directory
# by setting up the correct paths
APP_DIR = Path(__file__).parent
os.chdir(APP_DIR)  # Change working directory to project root

# Add the project directory to Python path for imports to work correctly
sys.path.insert(0, str(APP_DIR))

# Import the main function from the app implementation
from app.main import main

# When this file is executed directly, run the main function
if __name__ == "__main__":
    main()