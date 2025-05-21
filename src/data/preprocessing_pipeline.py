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

def clean_date_features(df):
    """
    Clean and convert date-related columns, and create new date features.
    
    Args:
        df: DataFrame with date columns
        
    Returns:
        DataFrame with cleaned date columns and new date features
    """
    df_copy = df.copy()
    
    # Convert Tranc_YearMonth into datetime if it's not already
    if 'Tranc_YearMonth' in df_copy.columns and not pd.api.types.is_datetime64_any_dtype(df_copy['Tranc_YearMonth']):
        df_copy['Tranc_YearMonth'] = pd.to_datetime(df_copy['Tranc_YearMonth'])
    
    # Create new date columns
    if 'Tranc_YearMonth' in df_copy.columns:
        date_features = ["year", "month", "quarter"]
        df_copy[date_features] = df_copy.apply(lambda row: pd.Series({
            "year": row.Tranc_YearMonth.year, 
            "month": row.Tranc_YearMonth.month, 
            "quarter": row.Tranc_YearMonth.quarter
        }), axis=1)
    
    # Create age at transaction feature
    if 'year' in df_copy.columns and 'lease_commence_date' in df_copy.columns:
        df_copy['age_at_tranc'] = df_copy['year'] - df_copy['lease_commence_date']
    
    # Convert datatypes to object
    for col in ['bus_interchange', 'mrt_interchange', 'pri_sch_affiliation', 'affiliation']:
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
    return [
        'town', 'flat_type', 'storey_range', 'floor_area_sqm', 'flat_model', 
        'lease_commence_date', 'Tranc_Year', 'Tranc_Month', 'lower', 'upper', 'mid', 
        'hdb_age', 'address', 'year_completed', 'residential', 'postal',
        'Latitude', 'Longitude', 'mrt_latitude', 'mrt_longitude', 'bus_stop_name', 
        'bus_stop_latitude', 'bus_stop_longitude', 'pri_sch_latitude', 'pri_sch_longitude', 
        'sec_sch_latitude', 'sec_sch_longitude', 'Tranc_YearMonth', 'block'
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
    
    # Clean date features
    df_copy = clean_date_features(df_copy)
    
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