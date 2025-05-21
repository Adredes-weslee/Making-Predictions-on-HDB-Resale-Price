"""Module for preprocessing HDB resale data.

This module provides functions to clean, transform, and prepare raw HDB resale
transaction data for analysis and modeling. It handles data cleaning tasks like
missing value imputation, data type conversion, and outlier treatment.

⚠️ IMPORTANT NOTE FOR MODEL TRAINING ⚠️
For model training and production deployment, please use the pipeline-based approach
implemented in src.models.training instead of these direct function calls.
This module is primarily maintained for exploratory data analysis, visualization,
and backward compatibility.

The preprocessing pipeline includes:
1. Cleaning raw data (handling missing values, correcting data types)
2. Creating temporal features from transaction dates
3. Calculating property age and remaining lease features
4. Creating location-based features
5. Normalizing and transforming numerical features
6. Encoding categorical features

Each preprocessing step is implemented as a separate function to allow for
flexible pipeline construction and easier testing.

Typical usage:
    >>> from src.data.preprocessing import preprocess_data
    >>> import pandas as pd
    >>> raw_df = pd.read_csv("data/raw/train.csv")
    >>> processed_df = preprocess_data(raw_df)
"""
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw HDB data by handling missing values and data types.
    
    This function performs initial cleaning on raw HDB resale data by:
    1. Converting date columns to datetime format
    2. Converting categorical columns to appropriate data types
    3. Handling any basic data quality issues
    
    The function creates a copy of the input DataFrame to avoid modifying
    the original data, following best practices for data processing.
    
    Args:
        df (pd.DataFrame): Raw dataframe containing HDB resale transaction data.
            Expected to have columns like 'Tranc_YearMonth', 'bus_interchange', etc.
        
    Returns:
        pd.DataFrame: A cleaned copy of the input dataframe with proper data types
            and basic data quality issues addressed.
            
    Raises:
        No exceptions are explicitly raised, but pandas operations may raise
        exceptions for invalid data.
        
    Example:
        >>> raw_df = pd.read_csv("data/raw/train.csv")
        >>> cleaned_df = clean_data(raw_df)
        >>> cleaned_df.dtypes['Tranc_YearMonth']
        datetime64[ns]
    """
    df = df.copy()
    
    # Convert date columns
    if 'Tranc_YearMonth' in df.columns:
        df['Tranc_YearMonth'] = pd.to_datetime(df['Tranc_YearMonth'])
    
    # Convert categorical columns
    for col in ['bus_interchange', 'mrt_interchange', 'pri_sch_affiliation', 'affiliation']:
        if col in df.columns:
            df[col] = df[col].astype('category')
    
    return df


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
            
    Raises:
        AttributeError: If 'Tranc_YearMonth' exists but is not a datetime column.
        
    Example:
        >>> df = pd.DataFrame({'Tranc_YearMonth': pd.to_datetime(['2019-01', '2019-04'])})
        >>> df_with_features = create_date_features(df)
        >>> print(df_with_features)
          Tranc_YearMonth  year  month  quarter
        0      2019-01-01  2019      1        1
        1      2019-04-01  2019      4        2
    """
    df = df.copy()
    
    if 'Tranc_YearMonth' in df.columns:
        df['year'] = df['Tranc_YearMonth'].dt.year
        df['month'] = df['Tranc_YearMonth'].dt.month
        df['quarter'] = df['Tranc_YearMonth'].dt.quarter
    
    return df


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
            
    Raises:
        KeyError: If required columns are missing from the DataFrame.
        
    Example:
        >>> df = pd.DataFrame({
        ...     'year': [2019, 2019, 2019],
        ...     'lease_commence_date': [1980, 1990, 2000]
        ... })
        >>> df_with_age = create_age_features(df)
        >>> df_with_age['building_age'].tolist()
        [39, 29, 19]
    """
    df = df.copy()
    
    if 'year' in df.columns and 'lease_commence_date' in df.columns:
        # Calculate building age
        df['building_age'] = df['year'] - df['lease_commence_date']
        
        # Calculate remaining lease
        df['remaining_lease'] = 99 - df['building_age']
        
        # Handle negative remaining lease (shouldn't happen, but just in case)
        df['remaining_lease'] = df['remaining_lease'].clip(lower=0)
        
        # Create lease decay feature (non-linear transformation)
        # The impact of lease decay accelerates as the lease gets shorter
        df['lease_decay'] = 1 / (df['remaining_lease'] + 1)
    
    return df


def get_columns_to_drop() -> List[str]:
    """Get the list of columns to drop during preprocessing.
    
    This function returns a predefined list of columns that should be dropped from
    the dataset during preprocessing. Columns are dropped for various reasons:
    
    1. Redundant columns (e.g., when their information is captured by derived features)
    2. Columns with too many unique values that would create excessive dimensionality
    3. Raw columns that have been transformed into more useful features
    4. Columns that leak future information or are not available at prediction time
    5. Identifier columns with no predictive value
    
    This function centralizes the column dropping logic to ensure consistency
    across different preprocessing runs.
    
    Returns:
        List[str]: List of column names to drop during preprocessing.
        
    Example:
        >>> cols_to_drop = get_columns_to_drop()
        >>> preprocessed_df = raw_df.drop(columns=[c for c in cols_to_drop if c in raw_df.columns])
    """
    return [
        'town', 'flat_type', 'storey_range', 'floor_area_sqm', 'flat_model', 'lease_commence_date', 'Tranc_Year', 
        'Tranc_Month', 'lower', 'upper', 'mid', 'hdb_age', 'address', 'year_completed', 'residential', 'postal',
        'Latitude', 'Longitude', 'mrt_latitude', 'mrt_longitude', 'bus_stop_name', 'bus_stop_latitude', 
        'bus_stop_longitude', 'pri_sch_latitude', 'pri_sch_longitude', 'sec_sch_latitude', 'sec_sch_longitude',
        'Tranc_YearMonth', 'block'
    ]


def get_missing_value_columns() -> List[str]:
    """Get the list of columns with significant missing values.
    
    This function returns a predefined list of columns that are known to have
    a significant percentage of missing values in the HDB dataset. These columns
    typically represent amenity proximity features (e.g., distance to malls, hawker centers)
    that may be missing for certain locations.
    
    The identified columns can be:
    1. Dropped to simplify the model at the cost of losing potentially useful information
    2. Imputed using various techniques (mean, median, KNN, etc.)
    3. Treated specially with dedicated missing value indicators
    
    Returns:
        List[str]: List of column names that have significant missing values.
        
    Example:
        >>> missing_cols = get_missing_value_columns()
        >>> # To drop these columns:
        >>> clean_df = df.drop(columns=[c for c in missing_cols if c in df.columns])
        >>> # Or to impute:
        >>> from sklearn.impute import SimpleImputer
        >>> imputer = SimpleImputer(strategy='median')
        >>> df[missing_cols] = imputer.fit_transform(df[missing_cols])
    """
    return [
        'Mall_Within_500m', 'Mall_Within_1km', 'Hawker_Within_500m', 'Hawker_Within_1km', 'Hawker_Within_2km'
    ]


def preprocess_data(
    df: pd.DataFrame, 
    drop_cols: Optional[List[str]] = None,
    drop_missing_cols: bool = True,
    is_training: bool = True
) -> Union[Tuple[pd.DataFrame, pd.Series], pd.DataFrame]:
    """Preprocess the HDB resale data for model training or prediction.
    
    This function serves as the main preprocessing pipeline for HDB resale data.
    It performs a sequence of data preparation steps:
    
    1. Basic cleaning (data types, missing values)
    2. Feature creation (temporal features, age features)
    3. Column dropping (based on feature selection)
    4. Missing value handling (dropping or imputation)
    5. Training/test set preparation
    
    The function handles both training and prediction scenarios. For training data,
    it separates the target variable. For test data, it ensures the same preprocessing
    steps are applied consistently.
    
    Args:
        df (pd.DataFrame): Raw dataframe to preprocess, containing HDB resale data.
        drop_cols (List[str], optional): List of columns to drop. If None, uses the 
            default list from get_columns_to_drop(). Defaults to None.
        drop_missing_cols (bool): Whether to drop columns with significant missing values.
            If True, columns in get_missing_value_columns() will be dropped.
            If False, these columns are retained (missing values should be handled later).
            Defaults to True.
        is_training (bool): Whether this is the training dataset. If True, separates
            and returns the target variable 'resale_price'. Defaults to True.
        
    Returns:
        If is_training:
            Tuple[pd.DataFrame, pd.Series]: Features DataFrame X and target Series y.
        Else:
            pd.DataFrame: Features DataFrame X.
            
    Raises:
        ValueError: If 'resale_price' column is missing in training data.
        
    Example:
        >>> # For training data:
        >>> X_train, y_train = preprocess_data(train_df, is_training=True)
        >>> # For test data:
        >>> X_test = preprocess_data(test_df, is_training=False)
    """
    df = df.copy()
    
    # Apply basic cleaning
    df = clean_data(df)
    
    # Create additional features
    df = create_date_features(df)
    df = create_age_features(df)
    
    # Get columns to drop
    if drop_cols is None:
        drop_cols = get_columns_to_drop()
    
    # Drop columns
    existing_cols = [col for col in drop_cols if col in df.columns]
    if existing_cols:
        df = df.drop(columns=existing_cols)
    
    # Drop columns with significant missing values
    if drop_missing_cols:
        missing_cols = get_missing_value_columns()
        existing_missing_cols = [col for col in missing_cols if col in df.columns]
        if existing_missing_cols:
            df = df.drop(columns=existing_missing_cols)
    
    # Split target and features for training data
    if is_training:
        if 'resale_price' not in df.columns:
            raise ValueError("Target column 'resale_price' not found in training data")
        y = df['resale_price']
        X = df.drop(columns=['resale_price'])
        return X, y
    else:
        return df
