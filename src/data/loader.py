"""Module for loading and initial processing of HDB resale data.

This module provides utility functions for loading, accessing, and managing
the HDB resale property dataset. It handles the access path resolution for different
data files (raw, processed), splits data into training and testing sets, and provides
standardized interfaces for other components to access the data.

The module abstracts the details of file locations and data loading operations, 
providing a consistent interface regardless of the underlying file structure.

Typical usage:
    >>> from src.data.loader import get_data_paths, load_raw_data
    >>> paths = get_data_paths()
    >>> df = load_raw_data(paths['train'])
"""
import os
from typing import Dict, Optional, Tuple, Union

import pandas as pd
from sklearn.model_selection import train_test_split


def load_raw_data(file_path: str) -> pd.DataFrame:
    """Load raw HDB data from CSV file.
    
    This function reads a CSV file containing HDB resale transaction data and
    loads it into a pandas DataFrame. It uses low_memory=False to ensure
    correct data type inference when dealing with mixed-type columns.
    
    Args:
        file_path (str): Path to the CSV file containing the raw data.
        
    Returns:
        pd.DataFrame: Loaded data as a pandas DataFrame with all columns as
            found in the original CSV.
            
    Raises:
        FileNotFoundError: If the specified file path does not exist.
        pd.errors.EmptyDataError: If the CSV file is empty.
        pd.errors.ParserError: If the CSV file is malformed.
    
    Example:
        >>> df = load_raw_data("data/raw/hdb_transactions.csv")
        >>> print(df.shape)
        (60000, 40)
    """
    return pd.read_csv(file_path, low_memory=False)


def get_data_paths(base_dir: Optional[str] = None) -> Dict[str, str]:
    """Get paths to data files used in the project.
    
    This function resolves the absolute paths to all data files used in the project,
    including raw training and test data, as well as processed data files. It uses
    a standardized directory structure and can work with either a provided base directory
    or automatically detect the project root.
    
    The function assumes the following directory structure:
    project_root/
    ├── data/
    │   ├── raw/
    │   │   ├── train.csv
    │   │   └── test.csv
    │   └── processed/
    │       └── kaggle_hdb_df.csv
    
    Args:
        base_dir (str, optional): Base directory of the project. If None, the function
            will attempt to detect the project root automatically by traversing up from
            the current module's location.
        
    Returns:
        Dict[str, str]: Dictionary with keys 'train', 'test', and 'processed', each
            mapping to the absolute path of the respective data file.
    
    Example:
        >>> paths = get_data_paths()
        >>> print(paths['train'])
        '/path/to/project/data/raw/train.csv'
    """
    if base_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    return {
        "train": os.path.join(base_dir, "data", "raw", "train.csv"),
        "test": os.path.join(base_dir, "data", "raw", "test.csv"),
        "processed": os.path.join(base_dir, "data", "processed", "kaggle_hdb_df.csv"),
    }


def load_train_test_data(
    test_size: float = 0.2, 
    random_state: int = 123
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load and split the HDB resale training data for model development.
    
    This function provides a convenient way to load the HDB resale training data
    and split it into features and target variables for both training and testing sets.
    The function handles the entire process of:
    1. Resolving the path to the training data file
    2. Loading the raw data from CSV
    3. Separating features from the target variable (resale_price)
    4. Splitting the data into training and test sets
    
    This consolidated function eliminates the need for calling multiple data loading
    and processing functions separately, streamlining the model development workflow.
    
    Args:
        test_size (float, optional): Proportion of the dataset to include in the test
            split. Should be between 0.0 and 1.0. Defaults to 0.2 (20% test).
        random_state (int, optional): Random seed used by the train_test_split function
            to ensure reproducibility of the data split. Defaults to 123.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: A tuple containing four
            elements in the following order:
            - X_train: DataFrame with feature variables for training
            - X_test: DataFrame with feature variables for testing
            - y_train: Series with target variable (resale_price) for training
            - y_test: Series with target variable (resale_price) for testing
            
    Raises:
        FileNotFoundError: If the training data file cannot be found.
        KeyError: If the 'resale_price' column is missing from the loaded data.
            
    Example:
        >>> X_train, X_test, y_train, y_test = load_train_test_data(test_size=0.25)
        >>> print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
        Training samples: 45000, Test samples: 15000
        >>> print(f"Average training price: ${y_train.mean():.2f}")
        Average training price: $450000.00
    """
    data_paths = get_data_paths()
    df = load_raw_data(data_paths["train"])
    
    # Assume 'resale_price' is the target variable
    y = df["resale_price"]
    X = df.drop(columns=["resale_price"])
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def load_test_data() -> pd.DataFrame:
    """Load the HDB resale test data for making predictions.
    
    This function provides a simple interface to load the test dataset that is used for
    making predictions on unseen data. It handles the path resolution and loading process,
    ensuring that the test data is loaded consistently across different parts of the application.
    
    The test data is expected to have the same feature columns as the training data, but
    without the target variable ('resale_price').
    
    Returns:
        pd.DataFrame: The test dataset with all feature columns, ready for preprocessing
            and prediction.
            
    Raises:
        FileNotFoundError: If the test data file cannot be found.
        pd.errors.EmptyDataError: If the test data file is empty.
        pd.errors.ParserError: If the test data file is malformed.
            
    Example:
        >>> test_df = load_test_data()
        >>> print(f"Loaded {test_df.shape[0]} test samples with {test_df.shape[1]} features")
    """
    data_paths = get_data_paths()
    return load_raw_data(data_paths["test"])


def save_processed_data(df: pd.DataFrame, filename: str) -> None:
    """Save processed data to a CSV file in the processed data directory.
    
    This function saves a processed DataFrame to the project's processed data directory
    with the specified filename. It handles the path resolution to ensure the file is saved
    in the correct location regardless of where the function is called from.
    
    Saving processed data is useful to:
    1. Cache preprocessing results to avoid redundant computation
    2. Share processed datasets between different parts of the project
    3. Create checkpoints in multi-stage data processing pipelines
    4. Allow for manual inspection of the processed data
    
    Args:
        df (pd.DataFrame): The processed DataFrame to save.
        filename (str): Name of the file to save (without path). The file will be
            saved in the project's data/processed directory.
            
    Returns:
        None: The function saves the file but doesn't return anything.
        
    Example:
        >>> processed_df = preprocess_data(raw_df)
        >>> save_processed_data(processed_df, "processed_train_data.csv")
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_path = os.path.join(base_dir, "data", "processed", filename)
    df.to_csv(output_path, index=False)
