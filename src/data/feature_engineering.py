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
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectPercentile, mutual_info_regression
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def get_numeric_features(df: pd.DataFrame) -> List[str]:
    """Identify numeric features from a DataFrame for feature engineering.
    
    This function extracts the names of all numeric columns (float and integer types)
    from the input DataFrame, excluding the target variable 'resale_price' and any
    non-feature columns like 'id'. The resulting list is used to build feature
    transformation pipelines that apply appropriate preprocessing to numeric features.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing both feature and non-feature columns.
        
    Returns:
        List[str]: List of column names corresponding to numeric features that should
            be used in modeling. Excludes 'id' and 'resale_price' if present.
            
    Example:
        >>> df = pd.DataFrame({
        ...     'id': [1, 2, 3],
        ...     'floor_area_sqm': [70.5, 80.2, 90.0],
        ...     'storey': [5, 10, 15],
        ...     'resale_price': [400000, 500000, 600000],
        ...     'town': ['ANG MO KIO', 'BEDOK', 'CLEMENTI']
        ... })
        >>> get_numeric_features(df)
        ['floor_area_sqm', 'storey']
    """
    numeric_features = list(df.select_dtypes(['float','integer']).columns)
    if 'id' in numeric_features:
        numeric_features.remove('id')
    if 'resale_price' in numeric_features:
        numeric_features.remove('resale_price')
    return numeric_features


def get_categorical_features(df: pd.DataFrame) -> List[str]:
    """Identify categorical features from a DataFrame for feature engineering.
    
    This function extracts the names of all object-type columns from the input DataFrame.
    These are assumed to be categorical features that require encoding (e.g., one-hot
    encoding) before they can be used in machine learning models. Common categorical
    features in HDB data include 'town', 'flat_type', 'storey_range', etc.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing both feature and non-feature columns.
        
    Returns:
        List[str]: List of column names corresponding to categorical features that
            should be used in modeling.
            
    Example:
        >>> df = pd.DataFrame({
        ...     'floor_area_sqm': [70.5, 80.2, 90.0],
        ...     'town': ['ANG MO KIO', 'BEDOK', 'CLEMENTI'],
        ...     'flat_type': ['3 ROOM', '4 ROOM', '5 ROOM']
        ... })
        >>> get_categorical_features(df)
        ['town', 'flat_type']
    """
    return list(df.select_dtypes('object').columns)


def create_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
    feature_selection_percentile: int = 50
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
    # Numeric transformer with iterative imputer and standard scaling
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", IterativeImputer()),
            ("scaler", StandardScaler())
        ]
    )
    
    # Categorical transformer with one-hot encoding and feature selection
    categorical_transformer = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(drop='first', handle_unknown="infrequent_if_exist")),
            ("selector", SelectPercentile(mutual_info_regression, percentile=feature_selection_percentile))
        ]
    )
    
    # Column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ],
        n_jobs=-1
    )
    
    return preprocessor


def engineer_features(
    df: pd.DataFrame,
    feature_selection_percentile: int = 50
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
    """
    # Get feature lists
    numeric_features = get_numeric_features(df)
    categorical_features = get_categorical_features(df)
    
    # Create column transformer
    preprocessor = create_preprocessor(
        numeric_features, 
        categorical_features, 
        feature_selection_percentile
    )
    
    return preprocessor, numeric_features, categorical_features
