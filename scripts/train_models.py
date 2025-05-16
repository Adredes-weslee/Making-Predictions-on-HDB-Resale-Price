"""Script for training and evaluating machine learning models for HDB resale price prediction.

This script handles the end-to-end model training workflow for the HDB resale price
prediction project. It loads processed data, trains various regression models, evaluates 
their performance, and saves the trained models to disk for later use.

The script supports training different types of regression models:
1. Linear Regression - Standard OLS regression
2. Lasso Regression - L1 regularized regression
3. Ridge Regression - L2 regularized regression

For each model, the script:
1. Trains the model on the training data
2. Evaluates performance on both training and test data
3. Reports metrics including RMSE and R² score
4. Serializes the model for future use

The script can be run with command-line arguments to specify which models to train.

Typical usage:
    # Train all models
    $ python scripts/train_models.py
    
    # Train only linear regression model
    $ python scripts/train_models.py --linear
    
    # Train ridge and lasso models
    $ python scripts/train_models.py --ridge --lasso
"""
import os
import sys
import argparse
import pickle
from pathlib import Path

# Add the project root to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from src.data.loader import load_raw_data, get_data_paths
from src.data.preprocessing import preprocess_data
from src.data.feature_engineering import engineer_features
from src.models.linear import LinearRegressionModel, LassoRegressionModel, RidgeRegressionModel
from src.utils.helpers import ensure_dir, serialize_model


def train_and_evaluate_model(X_train, y_train, X_test, y_test, model_class, model_params=None):
    """Train a regression model and evaluate its performance on training and test data.
    
    This function handles the model training and evaluation workflow:
    1. Instantiates a model of the specified class with given parameters
    2. Fits the model on training data
    3. Generates predictions on both training and test data
    4. Computes and returns performance metrics
    
    The function is designed to work with any regression model class that follows
    the project's Model interface, allowing for consistent handling of different
    model types.
    
    Args:
        X_train (pd.DataFrame): Features for training the model.
        y_train (pd.Series): Target values for training the model.
        X_test (pd.DataFrame): Features for testing the model.
        y_test (pd.Series): Target values for testing the model.
        model_class (class): Python class of the model to instantiate.
        model_params (dict, optional): Parameters to pass to the model constructor.
            Defaults to None (empty dict).
        
    Returns:
        tuple: A tuple containing:
            - model (object): The trained model instance
            - metrics (dict): Dictionary of performance metrics including:
                - train_mse: Mean squared error on training data
                - test_mse: Mean squared error on test data
                - train_rmse: Root mean squared error on training data
                - test_rmse: Root mean squared error on test data
                - train_r2: R² score on training data
                - test_r2: R² score on test data
                
    Example:
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
        >>> model, metrics = train_and_evaluate_model(
        ...     X_train, y_train, X_test, y_test, LinearRegressionModel
        ... )
        >>> print(f"Test R²: {metrics['test_r2']:.4f}")
        >>> print(f"Test RMSE: {metrics['test_rmse']:.2f}")
    """
    # Initialize the model
    if model_params is None:
        model_params = {}
    
    model = model_class(**model_params)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Compute metrics
    metrics = {
        'train_mse': mean_squared_error(y_train, y_train_pred),
        'test_mse': mean_squared_error(y_test, y_test_pred),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'train_r2': r2_score(y_train, y_train_pred),
        'test_r2': r2_score(y_test, y_test_pred)
    }
    
    return model, metrics


def main(args):
    """Main execution function for the model training script.
    
    This function orchestrates the end-to-end model training workflow:
    1. Loads and preprocesses the training data
    2. Splits data into features (X) and target (y)
    3. Creates train/test split for evaluation
    4. Trains and evaluates specified regression models
    5. Saves trained models to disk
    
    The function's behavior is controlled by command-line arguments that specify
    which models to train. If no models are explicitly specified, all models are trained.
    
    Args:
        args (argparse.Namespace): Command-line arguments specifying which models to train.
            Includes boolean flags:
            - linear: Whether to train Linear Regression model
            - lasso: Whether to train Lasso Regression model
            - ridge: Whether to train Ridge Regression model
            - all: Whether to train all model types
    
    Returns:
        None: Results are printed to console and models are saved to disk.
        
    Example:
        >>> parser = argparse.ArgumentParser()
        >>> parser.add_argument('--linear', action='store_true')
        >>> args = parser.parse_args(['--linear'])
        >>> main(args)  # Trains only the linear regression model
    """
    # Load data
    data_paths = get_data_paths()
    df = load_raw_data(data_paths["train"])
    
    print(f"Loaded {len(df)} records")
    
    # Preprocess data
    processed_df = preprocess_data(df)
    
    print(f"Data preprocessed, {len(processed_df)} records remaining")
    
    # Engineer features
    featured_df = engineer_features(processed_df)
    
    print(f"Features engineered, {featured_df.shape[1]} features created")
    
    # Separate target variable
    y = featured_df['resale_price']
    X = featured_df.drop(columns=['resale_price'])
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Train linear regression model
    if args.linear or args.all:
        print("\nTraining Linear Regression model...")
        lr_model, lr_metrics = train_and_evaluate_model(
            X_train, y_train, X_test, y_test, LinearRegressionModel
        )
        
        print("Linear Regression metrics:")
        for metric, value in lr_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # Save the model
        serialize_model(lr_model, "linear_regression_model.pkl")
        print("Linear Regression model saved to 'models/linear_regression_model.pkl'")
    
    # Train Lasso regression model
    if args.lasso or args.all:
        print("\nTraining Lasso Regression model...")
        lasso_model, lasso_metrics = train_and_evaluate_model(
            X_train, y_train, X_test, y_test, LassoRegressionModel,
            model_params={'alpha': 0.01}
        )
        
        print("Lasso Regression metrics:")
        for metric, value in lasso_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # Save the model
        serialize_model(lasso_model, "lasso_regression_model.pkl")
        print("Lasso Regression model saved to 'models/lasso_regression_model.pkl'")
    
    # Train Ridge regression model
    if args.ridge or args.all:
        print("\nTraining Ridge Regression model...")
        ridge_model, ridge_metrics = train_and_evaluate_model(
            X_train, y_train, X_test, y_test, RidgeRegressionModel,
            model_params={'alpha': 1.0}
        )
        
        print("Ridge Regression metrics:")
        for metric, value in ridge_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # Save the model
        serialize_model(ridge_model, "ridge_regression_model.pkl")
        print("Ridge Regression model saved to 'models/ridge_regression_model.pkl'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ML models for HDB resale price prediction')
    parser.add_argument('--linear', action='store_true', help='Train Linear Regression model')
    parser.add_argument('--lasso', action='store_true', help='Train Lasso Regression model')
    parser.add_argument('--ridge', action='store_true', help='Train Ridge Regression model')
    parser.add_argument('--all', action='store_true', help='Train all models')
    
    args = parser.parse_args()
    
    # If no arguments provided, train all models
    if not any([args.linear, args.lasso, args.ridge, args.all]):
        args.all = True
    
    main(args)
