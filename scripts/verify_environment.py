"""
Environment Verification Script
==============================
This script verifies that the environment is properly set up for the HDB resale price
prediction analysis. It checks Python version, required packages, and directory structure
to ensure the project can run correctly.

Usage: 
    python verify_environment.py

The script will report on:
1. Python version compatibility
2. Required package installation status
3. Project directory structure
4. Availability of required model files
"""
import os
import sys
import importlib
import subprocess
from pathlib import Path

def check_python_version():
    """
    Check if Python version meets requirements
    
    Verifies that the current Python version is compatible with the project
    requirements. The project was developed with Python 3.8+, so this function
    warns if an older version is being used.
    """
    print(f"Python version: {sys.version}")
    major, minor, *_ = sys.version_info
    if major != 3 or minor < 8:
        print("⚠️ Warning: This project was developed with Python 3.8+. Some features may not work correctly.")
    else:
        print("✅ Python version: OK")

def check_required_packages():
    """
    Check if required packages are installed
    
    Attempts to import each required package and reports on whether
    it is installed and its version.
    """
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 'streamlit', 'plotly',
        'sklearn', 'pathlib', 'yaml', 'statsmodels', 'unittest'
    ]
    
    missing_packages = []
    
    print("\nChecking required packages:")
    for package in required_packages:
        try:
            if package == 'sklearn':
                module = importlib.import_module('sklearn')
                import sklearn
                version = getattr(sklearn, '__version__', 'unknown')
            else:
                module = importlib.import_module(package)
                version = getattr(module, '__version__', 'unknown')
            print(f"✅ {package}: {version}")
        except ImportError:
            print(f"❌ {package}: Not installed")
            missing_packages.append(package)
    
    if missing_packages:
        print("\n⚠️ Missing packages:")
        print("Run the following command to install them:")
        print(f"pip install {' '.join(missing_packages)}")
    else:
        print("\n✅ All required packages are installed")

def check_directory_structure():
    """
    Check if the directory structure is correct
    
    Verifies that all required directories for the project exist.
    Creates missing directories if they don't exist.
    """
    project_root = Path(__file__).parent.parent
    expected_dirs = [
        'app',
        'app/components', 
        'app/pages',
        'src',
        'src/data',
        'src/models',
        'src/visualization',
        'src/utils',
        'data',
        'data/raw',
        'data/processed',
        'models',
        'configs',
        'tests',
        'scripts',
        'notebooks'
    ]
    
    print("\nChecking directory structure:")
    for dir_path in expected_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"✅ {dir_path}: OK")
        else:
            print(f"❌ {dir_path}: Missing")
            try:
                full_path.mkdir(parents=True, exist_ok=True)
                print(f"  📂 Created {dir_path}")
            except Exception as e:
                print(f"  ⚠️ Could not create {dir_path}: {e}")

def check_config_files():
    """
    Check if the necessary configuration files exist
    
    Verifies that all required configuration files are present.
    """
    config_dir = Path(__file__).parent.parent / 'configs'
    required_files = [
        'app_config.yaml',
        'model_config.yaml'
    ]
    
    print("\nChecking configuration files:")
    for file in required_files:
        file_path = config_dir / file
        if file_path.exists():
            print(f"✅ {file}: OK")
        else:
            print(f"❌ {file}: Missing")

def check_model_files():
    """
    Check if the necessary model files exist
    
    Verifies that at least some model files are present.
    """
    model_dir = Path(__file__).parent.parent / 'models'
    
    print("\nChecking model files:")
    if not model_dir.exists():
        print(f"❌ Models directory does not exist")
        return
        
    model_files = list(model_dir.glob('*.pk1'))
    if model_files:
        print(f"✅ Found {len(model_files)} model files")
        for model in model_files[:5]:  # Show only first 5 models
            print(f"  - {model.name}")
        if len(model_files) > 5:
            print(f"  - ... and {len(model_files) - 5} more")
    else:
        print(f"❌ No model files found! You may need to train models first.")
        print("  Run 'python scripts/train_models.py' to train the models")

def check_streamlit():
    """
    Check if Streamlit is working correctly
    
    Verifies that Streamlit is installed and can be executed.
    """
    print("\nChecking Streamlit installation:")
    try:
        result = subprocess.run(
            ['streamlit', '--version'], 
            capture_output=True, 
            text=True
        )
        print(f"✅ Streamlit is installed: {result.stdout.strip()}")
    except Exception as e:
        print(f"❌ Error checking Streamlit: {e}")

def main():
    """
    Main function to run all checks
    """
    print("=" * 60)
    print("HDB Resale Price Prediction - Environment Verification")
    print("=" * 60)
    
    check_python_version()
    check_required_packages()
    check_directory_structure()
    check_config_files()
    check_model_files()
    check_streamlit()
    
    print("\n" + "=" * 60)
    print("Verification complete!")
    print("=" * 60)
    print("\nIf you need to generate processed data, run:")
    print("  python scripts/process_data.py")
    print("\nTo train models, run:")
    print("  python scripts/train_models.py")
    print("\nTo start the Streamlit application, run:")
    print("  streamlit run app/main.py")

if __name__ == "__main__":
    main()