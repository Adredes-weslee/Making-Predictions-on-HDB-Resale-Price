"""
Model Training Script
====================
This script trains machine learning models for HDB resale price prediction and
evaluates their performance. It supports multiple model types and hyperparameter
configurations specified in the model_config.yaml file.

The script leverages the preprocessed data created by preprocess_data.py to ensure
consistent feature engineering between training and prediction.

Usage:
    python scripts/train_models.py --model linear --evaluate
    python scripts/train_models.py --model [linear|ridge|lasso|all] [--evaluate] [--no-save]

Options:
    --model: Type of regression model to train (linear, ridge, lasso, or all)
    --evaluate: Evaluate model performance on test data
    --no-save: Run without saving the trained models
"""
import os
import sys
import argparse
import logging
import pickle
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# Add project root to path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

# Import project modules
from src.data.loader import get_data_paths, load_raw_data
from src.utils.helpers import load_config, setup_logging
from src.models.evaluation import evaluate_model
from src.models.base import get_model_class

# Setup logging
logger = logging.getLogger(__name__)
setup_logging(level=logging.INFO)

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning)

def load_processed_data():
    """
    Load preprocessed data for model training and testing.
    
    This function attempts to load the preprocessed data created by preprocess_data.py.
    If the preprocessed data isn't available, it logs an error and suggests running
    the preprocessing script first.
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test) if successful, None otherwise
    """
    logger.info("Loading preprocessed data for model training...")
    
    try:
        # Get data paths from configuration
        data_paths = get_data_paths()
        processed_dir = data_paths["processed"]
        
        # Define paths for preprocessed files
        train_path = os.path.join(processed_dir, "train_processed_exploratory.csv")
        test_path = os.path.join(processed_dir, "test_processed_exploratory.csv")
        
        # Check if preprocessed files exist
        if not os.path.exists(train_path):
            logger.error(f"Processed training data not found at {train_path}")
            logger.error("Please run 'python scripts/preprocess_data.py' first")
            return None
        
        # Load training data
        train_df = pd.read_csv(train_path)
        logger.info(f"Loaded training data with shape {train_df.shape}")
        
        # Extract target variable
        if 'resale_price' not in train_df.columns:
            logger.error("Target variable 'resale_price' not found in training data")
            return None
        
        y_train = train_df['resale_price']
        X_train = train_df.drop(columns=['resale_price'])
        
        # Load test data if available, otherwise create a test split
        if os.path.exists(test_path):
            test_df = pd.read_csv(test_path)
            logger.info(f"Loaded test data with shape {test_df.shape}")
            
            # Check if test data contains target variable
            if 'resale_price' in test_df.columns:
                y_test = test_df['resale_price']
                X_test = test_df.drop(columns=['resale_price'])
            else:
                logger.warning("Test data doesn't contain target variable. Using only for prediction.")
                X_test = test_df
                y_test = None
        else:
            logger.warning(f"Test data not found at {test_path}. Using 20% of training data for testing.")
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
        
        # Ensure feature alignment between train and test sets
        X_train, X_test = align_feature_columns(X_train, X_test)
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        logger.exception(f"Error loading processed data: {e}")
        return None

def align_feature_columns(X_train, X_test):
    """
    Ensure train and test sets have the same columns in the same order.
    
    Args:
        X_train: Training features DataFrame
        X_test: Testing features DataFrame
        
    Returns:
        tuple: (X_train, X_test) with aligned columns
    """
    logger.info("Aligning feature columns between train and test sets...")
    
    # Get common columns
    common_columns = list(set(X_train.columns) & set(X_test.columns))
    
    if len(common_columns) < len(X_train.columns):
        missing_in_test = set(X_train.columns) - set(X_test.columns)
        logger.warning(f"Test set is missing {len(missing_in_test)} columns that are in train set")
        
    if len(common_columns) < len(X_test.columns):
        extra_in_test = set(X_test.columns) - set(X_train.columns)
        logger.warning(f"Test set has {len(extra_in_test)} columns that are not in train set")
    
    # Use only common columns
    X_train_aligned = X_train[common_columns]
    X_test_aligned = X_test[common_columns]
    
    logger.info(f"Aligned datasets with {len(common_columns)} common features")
    return X_train_aligned, X_test_aligned

def load_model_config(model_name):
    """
    Load model hyperparameters from configuration.
    
    Args:
        model_name: Name of the model (linear, ridge, lasso)
        
    Returns:
        dict: Model hyperparameters
    """
    config = load_config('model_config')
    model_params = config.get('models', {}).get(f'{model_name}_regression', {})
    
    # Remove deprecated 'normalize' parameter for linear models in newer scikit-learn
    if 'normalize' in model_params:
        logger.warning("'normalize' parameter is deprecated in scikit-learn. Using StandardScaler instead.")
        model_params.pop('normalize')
        
    return model_params

def train_model(X_train, y_train, model_type):
    """
    Train a model with the specified type.
    
    Args:
        X_train: Training features
        y_train: Training target
        model_type: Type of model to train (linear, ridge, lasso)
        
    Returns:
        The trained model
    """
    logger.info(f"Training {model_type} regression model...")
    
    # Load model parameters from config
    model_params = load_model_config(model_type)
    logger.info(f"Model parameters: {model_params}")
    
    # Get the appropriate model class
    ModelClass = get_model_class(model_type)
    if ModelClass is None:
        logger.error(f"Unknown model type: {model_type}")
        return None
    
    # Create and train the model
    model = ModelClass(**model_params)
    model.fit(X_train, y_train)
    
    logger.info(f"{model_type.capitalize()} regression model trained successfully")
    return model

def save_model(model, model_type, X_train):
    """
    Save the trained model and feature information.
    
    Args:
        model: Trained model to save
        model_type: Type of model (used for filename)
        X_train: Training features used (for feature names)
    
    Returns:
        str: Path to the saved model file
    """
    # Create models directory if it doesn't exist
    model_dir = os.path.join(root_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    
    # Save the model
    model_path = os.path.join(model_dir, f"{model_type}_regression_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Save feature names
    feature_names = X_train.columns.tolist()
    feature_path = os.path.join(model_dir, f"{model_type}_regression_features.json")
    with open(feature_path, 'w', encoding='utf-8') as f:
        json.dump({
            'feature_names': feature_names,
            'feature_count': len(feature_names)
        }, f, indent=2)
    
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Feature information saved to {feature_path}")
    
    return model_path

def evaluate_and_report(model, X_test, y_test, model_type):
    """
    Evaluate a model and report performance metrics.
    
    Args:
        model: Trained model to evaluate
        X_test: Test features
        y_test: Test targets
        model_type: Name of the model type
        
    Returns:
        dict: Evaluation metrics
    """
    if y_test is None:
        logger.warning("No test labels available for evaluation")
        return None
        
    logger.info(f"Evaluating {model_type} model...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    # Create metrics report
    metrics = {
        'model_type': f"{model_type}_regression",
        'r2_score': r2,
        'rmse': rmse,
        'mae': mae
    }
    
    # Print metrics report
    logger.info(f"\n{model_type.upper()} REGRESSION EVALUATION:")
    logger.info(f"  R² Score: {r2:.4f}")
    logger.info(f"  RMSE: {rmse:.2f}")
    logger.info(f"  MAE: {mae:.2f}")
    
    # Save metrics to file
    metrics_dir = os.path.join(root_dir, "models")
    os.makedirs(metrics_dir, exist_ok=True)
    
    metrics_path = os.path.join(metrics_dir, f"{model_type}_regression_metrics.json")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"Metrics saved to {metrics_path}")
    return metrics

def main():
    """Main function for model training and evaluation."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train HDB resale price prediction models')
    parser.add_argument('--model', type=str, default='linear', 
                        choices=['linear', 'ridge', 'lasso', 'all'],
                        help='Type of regression model to train')
    parser.add_argument('--evaluate', action='store_true', 
                        help='Evaluate model on test data')
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save the trained model')
    
    args = parser.parse_args()
    
    # Determine which models to train
    models_to_train = ['linear', 'ridge', 'lasso'] if args.model == 'all' else [args.model]
    
    # Load data
    data = load_processed_data()
    if data is None:
        logger.error("Failed to load data. Exiting.")
        return
    
    X_train, X_test, y_train, y_test = data
    logger.info(f"Training data shape: {X_train.shape}")
    logger.info(f"Test data shape: {X_test.shape}")
    
    # Train specified models
    for model_type in models_to_train:
        # Train the model
        model = train_model(X_train, y_train, model_type)
        if model is None:
            logger.error(f"Failed to train {model_type} model. Skipping.")
            continue
        
        # Save the model if requested
        if not args.no_save:
            save_model(model, model_type, X_train)
        
        # Evaluate if requested
        if args.evaluate:
            evaluate_and_report(model, X_test, y_test, model_type)
    
    logger.info("Model training completed!")

if __name__ == "__main__":
    main()