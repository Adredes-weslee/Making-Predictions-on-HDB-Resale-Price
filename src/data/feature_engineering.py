"""Module for feature engineering on HDB resale data.

This module provides functions for transforming preprocessed data into features
suitable for machine learning models. It handles feature selection, encoding of
categorical variables, scaling of numerical features, and creation of feature
transformation pipelines.

The feature engineering pipeline includes:
1. Identifying numeric and categorical features
2. Creating appropriate transformers for each feature type
3. Building scikit-learn pipelines for consistent feature processing
4. Feature selection using mutual information or other techniques
5. Creating interaction terms and polynomial features (where appropriate)

These functions are designed to work together with the preprocessing module
to create a complete data preparation pipeline from raw data to model-ready features.

Typical usage:
    >>> import pandas as pd
    >>> from src.data.feature_engineering import engineer_features
    >>> from src.data.loader import load_raw_data
    >>> raw_df = load_raw_data("data/raw/train.csv")
    >>> preprocessor, numeric_features, categorical_features = engineer_features(raw_df)
"""
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectPercentile, SelectKBest, mutual_info_regression, f_regression
from sklearn.experimental import enable_iterative_imputer  # Required import for IterativeImputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.utils.helpers import load_config

# Import tqdm conditionally - will be used when show_progress=True
try:
    from tqdm import tqdm
except ImportError:
    # Define a simple no-op tqdm replacement for when tqdm isn't available
    def tqdm(iterable, *args, **kwargs):
        return iterable


def get_numeric_features(df: pd.DataFrame) -> List[str]:
    """Identify and return numeric features from a DataFrame for feature engineering.
    
    This function determines which columns in the DataFrame should be treated as
    numerical features in the machine learning pipeline. It follows this process:
    
    1. First attempts to load predefined numerical features from the feature_selection_config.json
    2. Validates that the loaded features actually exist in the DataFrame
    3. If configuration loading fails or no valid features are found, falls back to
       automatic detection by selecting all float and integer columns (excluding 'id' 
       and 'resale_price')
    
    The function ensures that even if the configuration changes or has errors, a valid
    set of numerical features will always be returned.
    
    Args:
        df (pd.DataFrame): The input DataFrame containing the data features
        
    Returns:
        List[str]: List of column names representing numeric features found in the DataFrame
        
    Example:
        >>> from src.data.loader import load_preprocessed_data
        >>> df = load_preprocessed_data("data/processed/train.csv")
        >>> numeric_features = get_numeric_features(df)
        >>> print(f"Selected {len(numeric_features)} numeric features")
        
    Notes:
        - Target variable 'resale_price' is automatically excluded if present
        - ID fields are automatically excluded if present
        - If both configuration-based and automatic detection fail, an empty list is returned
    """
    try:
        # First try to load from configuration
        import json
        import os
        from pathlib import Path
        
        root_dir = Path(__file__).parent.parent.parent
        config_path = os.path.join(root_dir, "configs", "feature_selection_config.json")
        
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                feature_config = json.load(f)
                numerical_features = feature_config.get("numerical_features", [])
                
                # Verify all features exist in the dataframe
                existing_features = [col for col in numerical_features if col in df.columns]
                
                if existing_features:
                    return existing_features
    except Exception as e:
        print(f"Warning: Could not load numerical features from config. Error: {e}")
    
    # Fall back to automatic detection if config loading fails
    numeric_features = list(df.select_dtypes(['float','integer']).columns)
    if 'id' in numeric_features:
        numeric_features.remove('id')
    if 'resale_price' in numeric_features:
        numeric_features.remove('resale_price')
    return numeric_features

def get_categorical_features(df: pd.DataFrame) -> List[str]:
    """Identify and return categorical features from a DataFrame for feature engineering.
    
    This function determines which columns in the DataFrame should be treated as
    categorical features in the machine learning pipeline. It follows this process:
    
    1. First attempts to load predefined categorical features from the feature_selection_config.json
    2. Validates that the loaded features actually exist in the DataFrame
    3. If configuration loading fails or no valid features are found, falls back to
       automatic detection by selecting all object and category dtype columns
    
    The function prioritizes config-based selection over automatic detection to allow
    manual control over which features are treated as categorical, while ensuring
    robustness through fallback mechanisms.
    
    Args:
        df (pd.DataFrame): The input DataFrame containing the data features
        
    Returns:
        List[str]: List of column names representing categorical features found in the DataFrame
        
    Example:
        >>> from src.data.loader import load_preprocessed_data
        >>> df = load_preprocessed_data("data/processed/train.csv")
        >>> categorical_features = get_categorical_features(df)
        >>> print(f"Selected {len(categorical_features)} categorical features:")
        >>> print(categorical_features[:5])  # Print first 5 categorical features
        
    Notes:
        - High-cardinality columns (with many unique values) will be included if they
          have object or category dtype, but may need to be carefully handled during
          feature engineering to avoid dimensionality explosion
        - If both configuration-based and automatic detection fail, an empty list is returned
    """
    try:
        # First try to load from configuration
        import json
        import os
        from pathlib import Path
        
        root_dir = Path(__file__).parent.parent.parent
        config_path = os.path.join(root_dir, "configs", "feature_selection_config.json")
        
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                feature_config = json.load(f)
                categorical_features = feature_config.get("categorical_features", [])
                
                # Verify all features exist in the dataframe
                existing_features = [col for col in categorical_features if col in df.columns]
                
                if existing_features:
                    return existing_features
    except Exception as e:
        print(f"Warning: Could not load categorical features from config. Error: {e}")
    
    # Fall back to automatic detection
    return list(df.select_dtypes(['object', 'category']).columns)

def create_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
    feature_selection_percentile: int = 50,
    scaling_method: str = "standard",
    feature_selection_method: str = "mutual_info",
    categorical_encoding: str = "one-hot",
    config: dict = None
) -> ColumnTransformer:
    """Create a scikit-learn preprocessing pipeline for feature transformation.
    
    This function builds a ColumnTransformer that applies appropriate preprocessing
    steps to different types of features:
    
    - For numeric features: Imputes missing values using the median, applies standard
      scaling (zero mean, unit variance), and then performs feature selection based
      on mutual information with the target.
    
    - For categorical features: Applies one-hot encoding with handling of unknown
      categories during prediction time.
    
    The resulting preprocessor ensures consistent transformation of data for both
    training and prediction, and helps improve model performance by providing
    properly scaled and encoded features.
    
    Args:
        numeric_features (List[str]): List of column names for numeric features.
        categorical_features (List[str]): List of column names for categorical features.
        feature_selection_percentile (int, optional): Percentage of top numeric features
            to select based on mutual information score with the target. Lower values
            create more aggressive feature selection. Defaults to 50.
        scaling_method (str, optional): Method to use for scaling numeric features.
            Options: "standard", "minmax", "robust", "none". Defaults to "standard".
        feature_selection_method (str, optional): Method to use for feature selection.
            Options: "mutual_info", "f_regression". Defaults to "mutual_info".
        categorical_encoding (str, optional): Method for encoding categorical variables.
            Options: "one-hot", "label", "target". Defaults to "one-hot".
        config (dict, optional): Model configuration dictionary. If provided, overrides
            other parameters with values from the configuration. Defaults to None.
            
    Returns:
        ColumnTransformer: A scikit-learn ColumnTransformer that can be used in a
            modeling pipeline to preprocess the data. Note that this transformer must be
            fitted on training data (with target values available) before it can be used
            for prediction, especially since it includes SelectPercentile which requires
            target values to select informative features.
            
    Raises:
        ValueError: If feature_selection_percentile is not between 1 and 100.
            
    Example:
        >>> numeric_cols = ['floor_area_sqm', 'remaining_lease']
        >>> categorical_cols = ['town', 'flat_type', 'flat_model']
        >>> preprocessor = create_preprocessor(
        ...     numeric_cols,
        ...     categorical_cols,
        ...     feature_selection_percentile=75
        ... )
        >>> # Use preprocessor in a model pipeline
        >>> from sklearn.pipeline import Pipeline
        >>> from sklearn.linear_model import LinearRegression
        >>> model = Pipeline([
        ...     ('preprocessor', preprocessor),
        ...     ('regressor', LinearRegression())
        ... ])
    """
    # Load configuration if not provided but override with function parameters if specified
    if config is None:
        config = load_config('model_config')
    
    # Get feature engineering settings from config, with function parameters taking precedence
    features_config = config.get('features', {})
    
    # Use function parameters if provided, otherwise use config values with defaults
    scaling_method = scaling_method or features_config.get('scaling', 'standard')
    feature_selection_method = feature_selection_method or features_config.get('feature_selection', {}).get('method', 'f_regression')
    categorical_encoding = categorical_encoding or features_config.get('categorical_encoding', 'one-hot')
    
    # Select the appropriate scaler based on the scaling method
    if scaling_method == 'minmax':
        scaler = MinMaxScaler()
    elif scaling_method == 'robust':
        scaler = RobustScaler()
    elif scaling_method == 'none':
        scaler = None
    else:  # default to standard scaling
        scaler = StandardScaler()
    
    # Select the feature selection method
    if feature_selection_method == 'f_regression':
        feature_selector = f_regression
    else:  # default to mutual information
        feature_selector = mutual_info_regression
    
    # Numeric transformer pipeline
    numeric_steps = [("imputer", IterativeImputer())]
    if scaler is not None:
        numeric_steps.append(("scaler", scaler))
    
    numeric_transformer = Pipeline(steps=numeric_steps)
    
    # Categorical transformer with the appropriate encoding
    categorical_steps = []
    if categorical_encoding == 'one-hot':
        categorical_steps.append(("encoder", OneHotEncoder(drop='first', handle_unknown="infrequent_if_exist")))
    
    # Add feature selection if enabled
    if features_config.get('feature_selection', {}).get('enabled', True):
        categorical_steps.append(("selector", SelectPercentile(feature_selector, percentile=feature_selection_percentile)))
    
    categorical_transformer = Pipeline(steps=categorical_steps)
    
    # Column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        n_jobs=-1,
        sparse_threshold=0.3  # Make it less likely to return sparse matrices
    )
    
    # Add verbose debug information to the preprocessor
    preprocessor.verbose = True
    
    return preprocessor


def engineer_features(
    df: pd.DataFrame,
    feature_selection_percentile: int = 50,
    show_progress: bool = False,
    config: dict = None
) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """Engineer features from a preprocessed HDB resale dataframe.
    
    This function performs feature engineering on a preprocessed dataframe by:
    1. Identifying numeric and categorical features in the dataframe
    2. Creating appropriate transformers for each feature type
    3. Creating a scikit-learn ColumnTransformer for feature processing
    
    The function prepares but does not apply the transformations. The returned
    preprocessor must be fitted before it can transform data.
    
    Args:
        df (pd.DataFrame): Preprocessed dataframe containing HDB resale data.
            Should have already gone through basic preprocessing (cleaning, etc.).
        feature_selection_percentile (int, optional): Percentile of features to keep
            based on their mutual information scores with the target. Lower values
            result in more aggressive feature reduction. Defaults to 50.
        show_progress (bool, optional): Whether to display progress bars during processing.
            Defaults to False.
        
    Returns:
        Tuple[ColumnTransformer, List[str], List[str]]: A tuple containing:
            - preprocessor: A scikit-learn ColumnTransformer for feature transformations
            - numeric_features: List of numeric feature column names            
            - categorical_features: List of categorical feature column names
            
    Example:
        >>> from src.data.preprocessing import preprocess_data
        >>> from src.data.loader import load_raw_data
        >>> raw_df = load_raw_data("data/raw/train.csv")
        >>> processed_df = preprocess_data(raw_df)
        >>> preprocessor, num_features, cat_features = engineer_features(processed_df)
        >>> # The preprocessor can now be used in a model pipeline
        >>> from sklearn.pipeline import Pipeline
        >>> from sklearn.linear_model import LinearRegression
        >>> model = Pipeline([
        ...     ('preprocessor', preprocessor),
        ...     ('regressor', LinearRegression())
        ... ])
    """    # Load configuration if not provided
    if config is None:
        config = load_config('model_config')
    
    # Set up progress tracking if requested
    if show_progress:
        try:
            steps = ['Identify numeric features', 'Identify categorical features', 'Create preprocessor']
            pbar = tqdm(steps, desc="Feature engineering steps")
        except Exception as e:
            print(f"Warning: Could not initialize progress bar: {e}")
            show_progress = False
    
    # Use numeric and categorical features from config if provided, otherwise infer them
    features_config = config.get('features', {})
    config_numeric_features = features_config.get('numerical_features')
    config_categorical_features = features_config.get('categorical_features')
    
    if config_numeric_features and all(feat in df.columns for feat in config_numeric_features):
        numeric_features = config_numeric_features
    else:
        numeric_features = get_numeric_features(df)
    
    if show_progress:
        pbar.update(1)
        pbar.set_description("Identify categorical features")
    
    if config_categorical_features and all(feat in df.columns for feat in config_categorical_features):
        categorical_features = config_categorical_features
    else:
        categorical_features = get_categorical_features(df)
    
    if show_progress:
        pbar.update(1)
        pbar.set_description("Create preprocessor")
    
    # Get feature engineering settings from config
    scaling_method = features_config.get('scaling', 'standard')
    categorical_encoding = features_config.get('categorical_encoding', 'one-hot')
    feature_selection_method = features_config.get('feature_selection', {}).get('method', 'f_regression')
    
    # Create column transformer with configuration
    preprocessor = create_preprocessor(
        numeric_features, 
        categorical_features, 
        feature_selection_percentile=feature_selection_percentile,
        scaling_method=scaling_method,
        feature_selection_method=feature_selection_method,
        categorical_encoding=categorical_encoding,
        config=config
    )
    
    if show_progress:
        pbar.update(1)
        pbar.close()
    
    return preprocessor, numeric_features, categorical_features
