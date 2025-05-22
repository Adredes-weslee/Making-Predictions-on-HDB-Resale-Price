"""
Preprocessing Pipeline Module
=============================
This module provides a standardized preprocessing pipeline for HDB resale data
that can be used across different model training scripts.

The pipeline implements data cleaning, feature engineering, and transformation
steps that ensure consistency between training and prediction.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.feature_selection import SelectPercentile, mutual_info_regression

def clean_date_columns(df):
    """
    Convert date columns to proper datetime types.
    
    Args:
        df: DataFrame with date columns
        
    Returns:
        DataFrame with properly typed date columns
    """
    df_copy = df.copy()
    
    # Convert Tranc_YearMonth into datetime if it's not already
    if 'Tranc_YearMonth' in df_copy.columns and not pd.api.types.is_datetime64_any_dtype(df_copy['Tranc_YearMonth']):
        df_copy['Tranc_YearMonth'] = pd.to_datetime(df_copy['Tranc_YearMonth'])
    
    return df_copy

def create_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create temporal features from the transaction date column.
    
    This function extracts useful time-based features from the transaction date:
    - Year: Captures long-term price trends and market cycles
    - Month: Captures seasonal patterns in the real estate market
    - Quarter: Captures quarterly market trends and reporting periods
    
    These features allow the model to learn temporal patterns in HDB resale prices.
    The function checks for the existence of the 'Tranc_YearMonth' column before
    attempting to extract features.
    
    Args:
        df (pd.DataFrame): DataFrame with 'Tranc_YearMonth' column of datetime type.
            If this column doesn't exist, the function returns the DataFrame unchanged.
        
    Returns:
        pd.DataFrame: A copy of the input DataFrame with additional columns:
            - 'year': The year of the transaction
            - 'month': The month of the transaction (1-12)
            - 'quarter': The quarter of the transaction (1-4)
    """
    df_copy = df.copy()
    
    if 'Tranc_YearMonth' in df_copy.columns:
        df_copy['year'] = df_copy['Tranc_YearMonth'].dt.year
        df_copy['month'] = df_copy['Tranc_YearMonth'].dt.month
        # df_copy['quarter'] = df_copy['Tranc_YearMonth'].dt.quarter
    
    return df_copy

def create_age_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create property age and remaining lease features.
    
    This function calculates important HDB flat age-related features:
    - building_age: Years since the flat was built (transaction year - lease commence year)
    - remaining_lease: Estimated years of lease remaining at transaction time
    - lease_decay: Non-linear transformation of remaining lease that better captures
      the accelerating impact of lease decay on property value
    
    HDB flats in Singapore typically have 99-year leases, and the remaining lease
    duration is known to significantly impact resale prices.
    
    Args:
        df (pd.DataFrame): DataFrame with 'year' column (transaction year) and
            'lease_commence_date' column (year when the flat's lease started).
        
    Returns:
        pd.DataFrame: A copy of the input DataFrame with additional columns:
            - 'building_age': Age of the building at transaction time
            - 'remaining_lease': Estimated remaining lease years
            - 'lease_decay': Non-linear transformation of remaining lease
    """
    df_copy = df.copy()
    
    if 'year' in df_copy.columns and 'lease_commence_date' in df_copy.columns:
        # Calculate building age
        df_copy['building_age'] = df_copy['year'] - df_copy['lease_commence_date']
        
        # Calculate remaining lease
        df_copy['remaining_lease'] = 99 - df_copy['building_age']
        
        # Handle negative remaining lease (shouldn't happen, but just in case)
        df_copy['remaining_lease'] = df_copy['remaining_lease'].clip(lower=0)
        
        # Create lease decay feature (non-linear transformation)
        # The impact of lease decay accelerates as the lease gets shorter
        df_copy['lease_decay'] = 1 / (df_copy['remaining_lease'] + 1)
    
    return df_copy

def convert_column_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert columns to appropriate data types for modeling.
    
    Args:
        df: DataFrame to process
        
    Returns:
        DataFrame with converted column types
    """
    df_copy = df.copy()
    
    # Convert these binary columns to object type for proper categorical handling
    binary_cols = ['bus_interchange', 'mrt_interchange', 'pri_sch_affiliation', 'affiliation']
    for col in binary_cols:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].astype('object')
    
    return df_copy

def get_columns_to_drop():
    """
    Return a list of columns to drop based on feature analysis.
    
    These columns are either redundant, too high-cardinality, or have been 
    replaced by engineered features.
    
    Returns:
        List of column names to drop
    """
    # Try to load the columns_to_drop from feature selection config
    try:
        import json
        import os
        from pathlib import Path
        
        root_dir = Path(__file__).parent.parent.parent
        config_path = os.path.join(root_dir, "configs", "feature_selection_config.json")
        
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                feature_config = json.load(f)
                return feature_config.get("columns_to_drop", [])
    except Exception as e:
        print(f"Warning: Could not load feature selection config. Using default columns to drop. Error: {e}")
    
    # Default list if config isn't available
    return [
        'id',
        'address',
        'postal',
        'block',
        'bus_stop_name',
        'sec_sch_name',
        'Latitude',
        'Longitude',
        'bus_stop_latitude', 
        'bus_stop_longitude',
        'pri_sch_latitude', 
        'pri_sch_longitude',
        'sec_sch_latitude', 
        'sec_sch_longitude',
        'mrt_latitude', 
        'mrt_longitude',
        'floor_area_sqm',
        'Tranc_YearMonth',
        # Add any additional columns you'd like to drop by default
    ]

def get_missing_value_columns():
    """
    Return lists of columns with missing values.
    
    Returns:
        Tuple containing:
            - List of columns with missing values that should be imputed
            - List of columns with missing values that should be dropped
    """
    # Columns to keep and impute
    keep_and_impute = ['Mall_Nearest_Distance', 'Mall_Within_2km']
    
    # Columns with too many missing values to impute effectively
    high_missing_columns = ['Mall_Within_500m', 'Mall_Within_1km', 'Hawker_Within_500m',
                           'Hawker_Within_1km', 'Hawker_Within_2km']
    
    return keep_and_impute, high_missing_columns

def create_standard_preprocessing_pipeline(categorical_features=None, numerical_features=None, 
                                          feature_percentile=50):
    """
    Create a standard preprocessing pipeline for HDB resale data.
    
    Args:
        categorical_features: List of categorical feature column names
        numerical_features: List of numerical feature column names
        feature_percentile: Percentage of features to select based on mutual information scores
        
    Returns:
        sklearn ColumnTransformer preprocessing pipeline
    """
    if numerical_features is None:
        numerical_features = []
    
    if categorical_features is None:
        categorical_features = []
    
    # Create numeric transformer with imputation and scaling
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", IterativeImputer(random_state=42)),
            ("scaler", StandardScaler())
        ]
    )

    # Create categorical transformer with one-hot encoding and feature selection
    categorical_transformer = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(drop='first', handle_unknown="infrequent_if_exist")),
            ("selector", SelectPercentile(mutual_info_regression, percentile=feature_percentile)),
        ]
    )
    
    # Combine transformers in a column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder='drop',  # Drop any columns not specified
        verbose_feature_names_out=False  # Keep feature names
    )
    
    return preprocessor

def prepare_data_for_modeling(df, is_training=True, drop_high_missing=True):
    """
    Prepare data for modeling by applying all preprocessing steps.
    
    Args:
        df: DataFrame to preprocess
        is_training: Whether this is training data (contains target variable)
        drop_high_missing: Whether to drop columns with high percentage of missing values
        
    Returns:
        Tuple containing:
            - Preprocessed X DataFrame
            - Target variable y Series (if is_training=True, otherwise None)
            - List of numerical features
            - List of categorical features
    """
    # Create a copy of the DataFrame
    df_copy = df.copy()
    
    # Clean and process date-related features
    df_copy = clean_date_columns(df_copy)
    df_copy = create_date_features(df_copy)
    df_copy = create_age_features(df_copy)
    df_copy = convert_column_types(df_copy)
    
    # Get columns to drop
    columns_to_drop = get_columns_to_drop()
    
    # Get missing value columns
    keep_and_impute, high_missing_columns = get_missing_value_columns()
    
    if drop_high_missing:
        columns_to_drop.extend(high_missing_columns)
        
    # Drop specified columns
    for col in columns_to_drop:
        if col in df_copy.columns:
            df_copy.drop(columns=col, inplace=True)
    
    # Extract target variable if this is training data
    y = None
    if is_training and 'resale_price' in df_copy.columns:
        y = df_copy['resale_price']
        df_copy.drop(columns=['resale_price'], inplace=True)
    
    # Remove ID column if present
    if 'id' in df_copy.columns:
        df_copy.drop(columns=['id'], inplace=True)
    
    # Identify numeric and categorical features
    numerical_features = list(df_copy.select_dtypes(['float', 'integer']).columns)
    categorical_features = list(df_copy.select_dtypes('object').columns)
    
    # Return prepared data
    return df_copy, y, numerical_features, categorical_features

def save_processed_data(X, y=None, output_path=None, save_y=True):
    """
    Save preprocessed data as CSV file for consistency.
    
    Args:
        X: Features DataFrame
        y: Target Series (optional)
        output_path: Path to save the CSV file
        save_y: Whether to include target variable in the saved file
    
    Returns:
        Path to the saved file
    """
    import os
    from pathlib import Path
    
    # Set default path if none provided
    if output_path is None:
        root_dir = Path(__file__).parent.parent.parent
        output_path = os.path.join(root_dir, "data", "processed", "train_pipeline_processed.csv")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Combine X and y if y is provided and save_y is True
    if y is not None and save_y:
        data = X.copy()
        data['resale_price'] = y
    else:
        data = X
    
    # Save to CSV
    data.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")
    
    return output_path