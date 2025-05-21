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

APP_DIR = Path(__file__).parent
APP_MODULE_DIR = APP_DIR / 'app'  # Add this line
os.chdir(APP_DIR)
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(APP_MODULE_DIR))  # Add this line

# Import the main function from the app implementation
from app.main import main

# When this file is executed directly, run the main function
if __name__ == "__main__":
    main()