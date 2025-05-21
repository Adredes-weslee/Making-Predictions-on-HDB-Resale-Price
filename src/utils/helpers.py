"""Helper functions for the HDB resale price prediction project.

This module provides utility functions that are used throughout the HDB resale 
price prediction project. These functions handle common tasks like:

1. Path resolution and directory management
2. Configuration loading and management
3. Data format conversion and transformation
4. File I/O operations for various formats (CSV, pickle, YAML)
5. Date/time operations specific to the project

The functions in this module are designed to be simple, reusable, and focused on
a single responsibility, following the Unix philosophy of "do one thing well."

Typical usage:
    >>> from src.utils.helpers import get_project_root, load_config
    >>> root_dir = get_project_root()
    >>> app_config = load_config('app_config')
    >>> print(f"App version: {app_config.get('version', '1.0.0')}")
"""
import os
import yaml
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import pickle


def get_project_root():
    """Get the absolute path to the project root directory.
    
    This function determines the absolute path to the root directory of the project
    by navigating up from the current file's location. It uses a relative path
    calculation based on the location of the helpers.py file within the project
    structure (src/utils/helpers.py).
    
    Using this function ensures that paths are resolved correctly regardless of the
    current working directory, making file operations more robust across different
    execution environments (development, testing, production).
    
    Returns:
        Path: A pathlib.Path object representing the absolute path to the project's
            root directory.
            
    Example:
        >>> root_dir = get_project_root()
        >>> data_dir = root_dir / 'data' / 'raw'
        >>> models_dir = root_dir / 'models'
    """
    current_file = Path(__file__).resolve()
    # Navigate up 2 directories from utils/ to reach project root
    return current_file.parent.parent.parent


def load_config(config_name):
    """Load configuration from a YAML file in the configs directory.
    
    This function loads configuration parameters from a specified YAML file in the
    project's configs directory. It resolves the path to the config file using
    the project root directory, handles the case where the file might not exist,
    and parses the YAML content into a Python dictionary.
    
    Configuration files are used to store parameters that might change between
    environments or runs, such as model hyperparameters, application settings,
    and feature engineering options.
    
    Args:
        config_name (str): Name of the configuration file without the .yaml extension.
            For example, 'app_config' for 'configs/app_config.yaml'.
        
    Returns:
        dict: A dictionary containing the configuration parameters from the YAML file.
            Returns an empty dictionary if the file doesn't exist.
            
    Example:
        >>> model_config = load_config('model_config')
        >>> learning_rate = model_config.get('learning_rate', 0.01)
    """
    config_path = get_project_root() / "configs" / f"{config_name}.yaml"
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
        return {}
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML configuration: {e}")
        return {}


def setup_logging(level=logging.INFO, log_file=None):
    """Set up logging configuration for the application.
    
    This function configures the Python logging system with appropriate formatters,
    handlers, and log levels. It creates a consistent logging environment across
    different modules in the application.
    
    Args:
        level (int, optional): The logging level to use. Should be one of the
            constants defined in the logging module (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            Defaults to logging.INFO.
        log_file (str, optional): Path to a file where logs should be written.
            If None, logs will only be sent to the console. Defaults to None.
    
    Returns:
        None: The function configures the logging system but doesn't return anything.
    
    Example:
        >>> from src.utils.helpers import setup_logging
        >>> import logging
        >>> # Basic setup with INFO level
        >>> setup_logging()
        >>> # Now you can use logging as normal
        >>> logger = logging.getLogger(__name__)
        >>> logger.info("Application started")
    """
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove any existing handlers to avoid duplicate logs
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Create file handler if log_file is specified
    if log_file:
        from pathlib import Path
        # Ensure log directory exists
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Suppress overly verbose logs from libraries
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    
def ensure_dir(directory):
    """Ensure that a directory exists, creating it if necessary.
    
    This utility function checks if a specified directory exists and creates it
    (including any necessary parent directories) if it doesn't. This is useful
    before writing files to ensure the target directory structure exists.
    
    The function handles both string paths and pathlib.Path objects, making it
    flexible to use with different path representations in the codebase.
    
    Args:
        directory (str or Path): The path to the directory that should exist.
            Can be either a string path or a pathlib.Path object.
            
    Returns:
        Path: A pathlib.Path object representing the directory path.
        
    Example:
        >>> from src.utils.helpers import ensure_dir
        >>> # Create output directory for model artifacts
        >>> model_output_dir = ensure_dir('outputs/model_results')
        >>> # Save a model visualization to this directory
        >>> plt.savefig(model_output_dir / 'feature_importance.png')
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def format_price(price):
    """Format a price value for display with Singapore dollar symbol.
    
    This function takes a numeric price value and formats it as a string with
    the Singapore dollar symbol (S$), thousands separator commas, and two decimal
    places. This consistent formatting improves readability in UI elements and reports.
    
    Args:
        price (float or int): The price value to format. Should be a numeric type
            that can be formatted with the :.2f format specifier.
        
    Returns:
        str: The formatted price string with currency symbol, e.g., "S$500,000.00"
        
    Example:
        >>> format_price(500000)
        'S$500,000.00'
        >>> format_price(1234.56)
        'S$1,234.56'
    """
    return f"S${price:,.2f}"


def calculate_remaining_lease(lease_start_year, current_year=None):
    """Calculate the remaining lease years for an HDB property.
    
    Singapore HDB flats typically have 99-year leases. This function calculates
    how many years remain on the lease based on when it started and the current year.
    The remaining lease duration is an important factor affecting resale prices.
    
    Args:
        lease_start_year (int): The year the lease commenced. This is when the
            99-year countdown began.
        current_year (int, optional): The current year or reference year for the
            calculation. If None, uses the current system year. Defaults to None.
        
    Returns:
        int: The number of years remaining on the lease (between 0 and 99).
            Returns 0 if the lease has already expired.
        
    Example:
        >>> # For a flat with lease starting in 1980, assuming current year is 2023:
        >>> calculate_remaining_lease(1980)
        56
        >>> # With an explicit reference year:
        >>> calculate_remaining_lease(1990, 2025)
        64
    """
    if current_year is None:
        current_year = datetime.now().year
        
    years_elapsed = current_year - lease_start_year
    return max(0, 99 - years_elapsed)


def serialize_model(model, filename):
    """Serialize a trained model to disk using pickle.
    
    This function saves a trained machine learning model to the project's models
    directory using pickle serialization. It ensures the models directory exists
    before attempting to save the file.
    
    Serializing models allows them to be:
    1. Saved after time-consuming training processes
    2. Shared between different components of the application
    3. Deployed to production environments
    4. Versioned and archived for future reference
    
    Args:
        model (object): The trained model object to serialize. This can be any
            Python object that can be pickled, such as scikit-learn models,
            pipelines, or custom model classes.
        filename (str): The name to give the serialized model file. This should
            include an appropriate extension, typically .pkl or .pickle.
        
    Returns:
        None: The function saves the model but doesn't return anything.
        
    Example:
        >>> from sklearn.linear_model import LinearRegression
        >>> model = LinearRegression()
        >>> model.fit(X_train, y_train)
        >>> serialize_model(model, "linear_regression_model.pkl")
    """
    model_dir = get_project_root() / "models"
    ensure_dir(model_dir)
    
    with open(model_dir / filename, 'wb') as f:
        pickle.dump(model, f)


def deserialize_model(filename):
    """Deserialize a model from disk using pickle.
    
    This function loads a previously serialized machine learning model from the
    project's models directory. It performs error handling to provide a clear
    message if the requested model file doesn't exist.
    
    Args:
        filename (str): The filename of the serialized model to load. This should
            be just the filename, not the full path, as the function will look in
            the standard models directory.
        
    Returns:
        object: The deserialized model object, ready to use for predictions or
            further training.
            
    Raises:
        FileNotFoundError: If the specified model file cannot be found in the
            models directory.
        
    Example:
        >>> try:
        ...     model = deserialize_model("linear_regression_model.pkl")
        ...     predictions = model.predict(X_test)
        ... except FileNotFoundError:
        ...     print("Model not found. Please train the model first.")
    """
    model_path = get_project_root() / "models" / filename
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    with open(model_path, 'rb') as f:
        return pickle.load(f)

