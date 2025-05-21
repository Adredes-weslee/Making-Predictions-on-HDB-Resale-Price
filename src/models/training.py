"""Training pipeline for HDB resale price prediction models.

This module provides functionality to train and save machine learning models for HDB
resale price prediction. It implements a comprehensive pipeline approach that:
1. Loads preprocessed training data
2. Defines preprocessing transformers for both numerical and categorical features
3. Creates and trains a full pipeline that includes both preprocessing and the model
4. Saves the entire pipeline for consistent prediction (preserving feature names)
5. Evaluates model performance and saves metrics

The pipeline approach ensures that the exact same preprocessing steps used during
training are automatically applied during prediction, maintaining consistency and
avoiding common errors related to feature transformation mismatches.

This module should be used to train new models or retrain existing models when new
data becomes available.

Example:
    >>> from src.models.training import train_and_save_pipeline_model
    >>> # Train a linear regression model
    >>> model_info = train_and_save_pipeline_model(
    ...     model_type='linear',
    ...     data_path='data/processed/train_processed.csv',
    ...     model_name='pipeline_linear_regression'
    ... )
    >>> print(f"Model R² score: {model_info['metrics']['r2']:.4f}")
"""
import os
import json
import pickle
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from src.data.preprocessing_pipeline import create_standard_preprocessing_pipeline

# Define model options
MODEL_TYPES = {
    'linear': LinearRegression,
    'ridge': Ridge,
    'lasso': Lasso
}

def load_training_data(data_path):
    """Load and prepare training data for model building.
    
    Args:
        data_path (str): Path to the CSV file containing training data.
        
    Returns:
        tuple: (X, y) where X is feature data and y is target variable.
        
    Raises:
        FileNotFoundError: If the data file doesn't exist.
        ValueError: If 'resale_price' column isn't found in the data.
    """
    try:
        # Load data
        df = pd.read_csv(data_path, encoding='utf-8')
        
        # Check if target variable exists
        if 'resale_price' not in df.columns:
            raise ValueError("Target variable 'resale_price' not found in dataset")
        
        # Debug: Print actual column names
        print(f"Loaded data columns: {df.columns[:10].tolist()}...")
        
        # Find columns with mixed types and convert them to strings
        for col in df.columns:
            if df[col].dtype == 'object':
                # Convert mixed-type columns to strings
                df[col] = df[col].astype(str)
        
        # Extract target
        y = df['resale_price']
        
        # Extract features (all columns except target)
        X = df.drop('resale_price', axis=1)
        
        return X, y
        
    except Exception as e:
        print(f"Error loading training data: {str(e)}")
        raise

def create_preprocessing_pipeline(X, categorical_features=None, numerical_features=None):
    """Train a full pipeline model and save it for consistent prediction.
    
    This function:
    1. Loads and prepares training data
    2. Creates preprocessing transformers for features
    3. Builds a full sklearn Pipeline with preprocessing and model steps
    4. Trains the pipeline on the data
    5. Evaluates model performance
    6. Saves the full pipeline and feature information
    
    Args:
        model_type (str): Type of model to train ('linear', 'ridge', or 'lasso').
        data_path (str): Path to the CSV file containing training data.
        model_name (str, optional): Name for the saved model files.
        model_dir (str, optional): Directory to save model files.
        test_size (float, optional): Proportion of data to use for test set.
        random_state (int, optional): Random seed for reproducibility.
        
    Returns:
        dict: Information about the trained model including performance metrics
              and paths to saved files.
              
    Raises:
        ValueError: If an invalid model_type is specified.
        FileNotFoundError: If data_path doesn't exist.
    """
    # If features aren't explicitly provided, auto-detect them
    if categorical_features is None:
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if numerical_features is None:
        numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Use the preprocessing pipeline from src.data.preprocessing_pipeline
    preprocessor = create_standard_preprocessing_pipeline(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        feature_percentile=99  # You can adjust this as needed
    )
    
    return preprocessor, categorical_features, numerical_features


def train_and_save_pipeline_model(model_type='linear', data_path=None, model_name=None, 
                                 model_dir=None, test_size=0.2, random_state=42):
    """Train a full pipeline model and save it for consistent prediction.
    
    This function:
    1. Loads and prepares training data
    2. Creates preprocessing transformers for features
    3. Builds a full sklearn Pipeline with preprocessing and model steps
    4. Trains the pipeline on the data
    5. Evaluates model performance
    6. Saves the full pipeline and feature information
    
    Args:
        model_type (str): Type of model to train ('linear', 'ridge', or 'lasso').
        data_path (str): Path to the CSV file containing training data.
        model_name (str, optional): Name for the saved model files.
        model_dir (str, optional): Directory to save model files.
        test_size (float, optional): Proportion of data to use for test set.
        random_state (int, optional): Random seed for reproducibility.
        
    Returns:
        dict: Information about the trained model including performance metrics
              and paths to saved files.
              
    Raises:
        ValueError: If an invalid model_type is specified.
        FileNotFoundError: If data_path doesn't exist.
    """
    # Get model class
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Invalid model_type. Must be one of: {', '.join(MODEL_TYPES.keys())}")
    
    ModelClass = MODEL_TYPES[model_type]
    
    # Set default paths if not provided
    if data_path is None:
        # Get project root directory
        root_dir = Path(__file__).parent.parent.parent
        data_path = os.path.join(root_dir, 'data', 'raw', 'train.csv')
    
    if model_dir is None:
        root_dir = Path(__file__).parent.parent.parent
        model_dir = os.path.join(root_dir, 'models')
    
    if model_name is None:
        model_name = f"pipeline_{model_type}_model"
    
    # Load training data
    X, y = load_training_data(data_path)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Create preprocessing pipeline
    preprocessor, categorical_features, numerical_features = create_preprocessing_pipeline(X)
    
    # Create full pipeline with preprocessing and model
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', ModelClass())
    ])
    
    # Train the model
    full_pipeline.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred_train = full_pipeline.predict(X_train)
    y_pred_test = full_pipeline.predict(X_test)
    
    metrics = {
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test))
    }
    
    # Save the pipeline
    os.makedirs(model_dir, exist_ok=True)
    pipeline_path = os.path.join(model_dir, f"{model_name}.pkl")
    joblib.dump(full_pipeline, pipeline_path)
    
    # Get transformed feature names (this is the key addition)
    try:
        # Get feature names from the pipeline
        feature_names = full_pipeline.named_steps['preprocessor'].get_feature_names_out()
        
        # Save feature information with real feature names
        feature_info = {
            'categorical_features': categorical_features,
            'numerical_features': numerical_features,
            'all_features': X.columns.tolist(),
            'transformed_features': feature_names.tolist()
        }
    except Exception as e:
        print(f"Warning: Could not get transformed feature names: {str(e)}")
        # Fall back to the previous method
        feature_info = {
            'categorical_features': categorical_features,
            'numerical_features': numerical_features,
            'all_features': X.columns.tolist()
        }
    
    feature_path = os.path.join(model_dir, f"{model_name}_features.json")
    with open(feature_path, 'w') as f:
        json.dump(feature_info, f)
    
    return {
        'metrics': metrics,
        'pipeline_path': pipeline_path,
        'feature_info_path': feature_path,
        'model_type': model_type,
        'feature_count': len(X.columns),
        'training_samples': len(X_train)
    }

if __name__ == "__main__":
    # Example usage
    model_info = train_and_save_pipeline_model(
        model_type='linear',
        model_name='pipeline_linear_regression'
    )
    
    print("Model training completed!")
    print(f"Training R²: {model_info['metrics']['train_r2']:.4f}")
    print(f"Test R²: {model_info['metrics']['test_r2']:.4f}")
    print(f"Model saved to: {model_info['pipeline_path']}")
