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
5. Availability of data files
"""
import sys
import importlib
from pathlib import Path

def check_python_version():
    """Check if the Python version is compatible with the project."""
    print("Checking Python version:")
    version = sys.version_info
    if version.major == 3 and version.minor >= 7:
        print(f"✅ Python version {version.major}.{version.minor}.{version.micro} is compatible")
    else:
        print(f"❌ Python version {version.major}.{version.minor}.{version.micro} is not compatible - please use Python 3.7+")

def check_required_packages():
    """Check if required packages are installed."""
    required_packages = [
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'sklearn',
        'streamlit',
        'plotly',
        'pyyaml'
    ]
    
    print("\nChecking required packages:")
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}: Not installed")
            missing_packages.append(package)
    
    if missing_packages:
        print("\nMissing packages. Install them with:")
        print(f"pip install {' '.join(missing_packages)}")

def check_directory_structure():
    """Check if the project directory structure is set up correctly."""
    base_dir = Path(__file__).parent.parent
    required_dirs = [
        'data',
        'data/raw',
        'data/processed',
        'models',
        'configs',
        'app',
        'src'
    ]
    
    print("\nChecking directory structure:")
    for dir_name in required_dirs:
        dir_path = base_dir / dir_name
        if dir_path.exists():
            print(f"✅ {dir_name}/")
        else:
            print(f"❌ {dir_name}/: Missing - creating...")
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"  ✓ Created {dir_name}/")

def check_config_files():
    """Check if configuration files exist."""
    base_dir = Path(__file__).parent.parent
    config_dir = base_dir / 'configs'
    required_files = [
        'app_config.yaml',
        'model_config.yaml'
    ]
    
    print("\nChecking configuration files:")
    for file in required_files:
        file_path = config_dir / file
        if file_path.exists():
            print(f"✅ {file}")
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
        
    model_files = list(model_dir.glob('*.pkl'))
    if model_files:
        print(f"✅ Found {len(model_files)} model files")
        for model in model_files[:5]:  # Show only first 5 models
            print(f"  - {model.name}")
        if len(model_files) > 5:
            print(f"  - ... and {len(model_files) - 5} more")
    else:
        print(f"❌ No model files found! You may need to train models first.")
        print("  Run 'python scripts/train_pipeline_model.py' to train the models")
    
    # Check for feature schema files
    schema_files = list(model_dir.glob('*.json'))
    if schema_files:
        print(f"✅ Found {len(schema_files)} feature schema files")
    else:
        print("⚠️ No feature schema files found. Feature name consistency may be an issue.")

def check_data_files():
    """
    Check if the necessary data files exist in data/raw directory
    """
    data_dir = Path(__file__).parent.parent / 'data' / 'raw'
    
    print("\nChecking data files:")
    if not data_dir.exists():
        print(f"❌ Data directory {data_dir} does not exist")
        return
        
    # Check for common data file formats
    data_files = list(data_dir.glob('*.csv'))
    data_files.extend(list(data_dir.glob('*.xlsx')))
    data_files.extend(list(data_dir.glob('*.parquet')))
    
    if data_files:
        print(f"✅ Found {len(data_files)} data files")
        for data_file in data_files[:5]:  # Show only first 5 files
            print(f"  - {data_file.name}")
        if len(data_files) > 5:
            print(f"  - ... and {len(data_files) - 5} more")
    else:
        print(f"❌ No data files found in {data_dir}")
        print("  Please add your raw data files to the data/raw directory")

def check_streamlit():
    """
    Check if Streamlit is working correctly
    
    Verifies that Streamlit is installed and can be executed.
    """
    print("\nChecking Streamlit installation:")
    try:
        import streamlit
        print(f"✅ Streamlit v{streamlit.__version__} is installed")
        
        # Check if the main app.py file exists
        app_path = Path(__file__).parent.parent / 'app.py'
        if app_path.exists():
            print(f"✅ Streamlit app file found: {app_path.name}")
            print("  Run with: streamlit run app.py")
        else:
            print(f"❌ Streamlit app file not found at {app_path}")
    except ImportError:
        print("❌ Streamlit is not installed")
        print("  Install with: pip install streamlit")

def main():
    """Run all environment verification checks."""
    print("=" * 60)
    print("HDB Resale Price Prediction - Environment Verification")
    print("=" * 60)
    
    check_python_version()
    check_required_packages()
    check_directory_structure()
    check_config_files()
    check_model_files()
    check_data_files()  # New function to check data files
    check_streamlit()
    
    print("\n" + "=" * 60)
    print("Verification complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()