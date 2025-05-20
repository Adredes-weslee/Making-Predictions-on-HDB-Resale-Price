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
from src.utils.helpers import ensure_dir, serialize_model, load_config


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


def create_dataframe_safely(feature_matrix, index, feature_names=None):
    """
    Create a DataFrame from a feature matrix with proper error handling to avoid
    array truth value ambiguity issues.
    
    Args:
        feature_matrix (numpy.array): Matrix of feature values
        index (pandas.Index): Index to use for the DataFrame
        feature_names (list): Names for columns, will create generic names if None
    
    Returns:
        pd.DataFrame: DataFrame with the given features
    """
    # Convert sparse matrix to dense if needed
    if hasattr(feature_matrix, 'toarray'):
        feature_matrix = feature_matrix.toarray()
    
    # If feature_names is None or doesn't match the column count, generate generic names
    feature_cols = feature_matrix.shape[1]
    if feature_names is None or len(feature_names) != feature_cols:
        feature_names = [f'feature_{i}' for i in range(feature_cols)]
    
    # Create DataFrame safely
    try:
        return pd.DataFrame(
            data=feature_matrix,
            index=index,
            columns=feature_names
        )
    except Exception as e:
        print(f"Error in DataFrame creation: {e}")
        # Fallback to a more basic approach
        df = pd.DataFrame(feature_matrix)
        df.index = index
        df.columns = feature_names
        return df


def process_raw_data(model_config, data_paths):
    """
    Process raw data through preprocessing and feature engineering steps.
    This mimics the functionality in preprocess_data.py to ensure consistency.
    
    Args:
        model_config (dict): Model configuration parameters
        data_paths (dict): Paths to data files
        
    Returns:
        tuple: A tuple containing (train_df, test_df) with processed data
    """
    print("Processing raw data...")
    
    # Load raw training data
    try:
        train_df = load_raw_data(data_paths["train"])
        print(f"Loaded {len(train_df)} training records from raw data")
    except FileNotFoundError:
        print(f"Raw training data file not found! Expected path: {data_paths['train']}")
        return None, None
    
    # Load raw test data
    test_df = None
    try:
        test_df = load_raw_data(data_paths["test"])
        print(f"Loaded {len(test_df)} test records from raw data")
    except FileNotFoundError:
        print(f"Raw test data file not found! Expected path: {data_paths['test']}")
    
    # Process training data
    print("Preprocessing training data...")
    try:
        # Preprocess data returns a tuple (X, y) for training data
        preprocessed_data = preprocess_data(train_df, is_training=True)
        
        # Check if it's a tuple and handle accordingly
        if isinstance(preprocessed_data, tuple):
            X_preprocessed, y_preprocessed = preprocessed_data
            # Create a DataFrame with both features and target for feature engineering
            train_preprocessed_df = X_preprocessed.copy()
            train_preprocessed_df['resale_price'] = y_preprocessed
        else:
            train_preprocessed_df = preprocessed_data
            
        print(f"Training data preprocessed, {len(train_preprocessed_df)} records remaining")
    except Exception as e:
        print(f"Error during training data preprocessing: {str(e)}")
        return None, None
    
    # Process test data if available
    test_preprocessed_df = None
    if test_df is not None:
        print("Preprocessing test data...")
        try:
            # For test data, preprocess_data returns a DataFrame directly
            test_preprocessed_df = preprocess_data(test_df, is_training=False)
            print(f"Test data preprocessed, {len(test_preprocessed_df)} records remaining")
        except Exception as e:
            print(f"Error during test data preprocessing: {str(e)}")
            # Continue with training data only
    
    # Extract feature configuration
    features_config = model_config.get('features', {})
    feature_selection_method = features_config.get('feature_selection', {}).get('method', 'f_regression')
    feature_selection_k_best = features_config.get('feature_selection', {}).get('k_best', 20)
    
    # Convert k_best to percentile if needed
    if isinstance(feature_selection_k_best, int) and feature_selection_k_best > 0 and feature_selection_k_best <= 100:
        feature_selection_percentile = feature_selection_k_best
    else:
        feature_selection_percentile = 50  # Default to 50 percentile
        
    print(f"Using feature selection: {feature_selection_method}, percentile: {feature_selection_percentile}")
    
    # Engineer features for training data
    print("Engineering features for training data...")
    try:
        # Get preprocessor with configuration
        preprocessor, numeric_features, categorical_features = engineer_features(
            train_preprocessed_df, 
            feature_selection_percentile=feature_selection_percentile,
            config=model_config  # Pass the entire config to the function
        )
        
        # Extract X and y for feature transformation
        X = train_preprocessed_df.drop('resale_price', axis=1, errors='ignore')
        y = train_preprocessed_df['resale_price']
        
        # Fit and transform with the target variable explicitly passed
        train_feature_matrix = preprocessor.fit_transform(X, y)
        
        # Create DataFrame with processed features
        train_featured_df = create_dataframe_safely(
            feature_matrix=train_feature_matrix,
            index=train_preprocessed_df.index,
            feature_names=None
        )
        
        # Add target back to DataFrame
        train_featured_df['resale_price'] = y
        
        print(f"Training features engineered, {train_featured_df.shape[1]} features created")
        
        # Save the processed training data
        processed_dir = os.path.dirname(data_paths['processed']) if os.path.isfile(data_paths['processed']) else data_paths['processed']
        os.makedirs(processed_dir, exist_ok=True)
        train_output_path = os.path.join(processed_dir, "train_processed.csv")
        train_featured_df.to_csv(train_output_path, index=False)
        print(f"Processed training data saved to: {train_output_path}")
        
    except Exception as e:
        print(f"Error during feature engineering for training data: {str(e)}")
        return None, None
    
    # Process test data if available
    test_featured_df = None
    if test_preprocessed_df is not None:
        print("Engineering features for test data...")
        try:
            # Transform test data using the already fitted preprocessor
            test_feature_matrix = preprocessor.transform(test_preprocessed_df)
            
            # Create DataFrame with same feature names as training
            test_featured_df = create_dataframe_safely(
                feature_matrix=test_feature_matrix,
                index=test_preprocessed_df.index,
                feature_names=None
            )
            
            # Add target if it exists in the original test data
            if 'resale_price' in test_df.columns:
                # We need to align indices between original test data and processed test data
                if all(idx in test_df.index for idx in test_preprocessed_df.index):
                    test_featured_df['resale_price'] = test_df.loc[test_preprocessed_df.index, 'resale_price']
            
            print(f"Test features engineered, {test_featured_df.shape[1]} features created")
            
            # Save the processed test data
            test_output_path = os.path.join(processed_dir, "test_processed.csv")
            test_featured_df.to_csv(test_output_path, index=False)
            print(f"Processed test data saved to: {test_output_path}")
            
        except Exception as e:
            print(f"Error during feature engineering for test data: {str(e)}")
            # Continue with training data only
    
    return train_featured_df, test_featured_df


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
    # Load model configuration
    model_config = load_config('model_config')
    
    # Get data paths
    data_paths = get_data_paths()
    
    # Get processed data directory
    processed_dir = data_paths["processed"]
    if os.path.isfile(processed_dir):
        processed_dir = os.path.dirname(processed_dir)
    
    # Define paths for processed files
    processed_train_path = os.path.join(processed_dir, "train_processed.csv")
    processed_test_path = os.path.join(processed_dir, "test_processed.csv")
    
    train_df = None
    test_df = None
    
    # Try to load processed training data
    if os.path.exists(processed_train_path):
        print(f"Loading processed training data from: {processed_train_path}")
        try:
            train_df = pd.read_csv(processed_train_path)
            print(f"Loaded processed training data with {train_df.shape[0]} records and {train_df.shape[1]} features")
            
            # Ensure 'resale_price' column exists in the loaded data
            if 'resale_price' not in train_df.columns:
                print("Warning: 'resale_price' column not found in processed training data. Cannot use this file.")
                train_df = None
        except Exception as e:
            print(f"Error loading processed training data: {str(e)}")
            train_df = None
    
    # Try to load processed test data
    if os.path.exists(processed_test_path):
        print(f"Loading processed test data from: {processed_test_path}")
        try:
            test_df = pd.read_csv(processed_test_path)
            print(f"Loaded processed test data with {test_df.shape[0]} records and {test_df.shape[1]} features")
            
            # Ensure it has the same columns as training data (if we have training data)
            if train_df is not None:
                if set(train_df.columns) != set(test_df.columns):
                    print("Warning: Column mismatch between training and test data.")
                    # Check if we can align the columns
                    common_cols = set(train_df.columns).intersection(set(test_df.columns))
                    if len(common_cols) > 0 and 'resale_price' in common_cols:
                        train_df = train_df[list(common_cols)]
                        test_df = test_df[list(common_cols)]
                        print(f"Using only {len(common_cols)} common columns between datasets.")
                    else:
                        print("Insufficient common columns, discarding processed test data.")
                        test_df = None
        except Exception as e:
            print(f"Error loading processed test data: {str(e)}")
            test_df = None
    
    # If either training or test data is missing, we need to process the raw data
    if train_df is None or test_df is None:
        train_df, test_df = process_raw_data(model_config, data_paths)
        
    # Ensure we have data to work with
    if train_df is None or 'resale_price' not in train_df.columns:
        print("Critical error: Failed to load or process training data.")
        return
    
    # Separate target variables
    X_train = train_df.drop('resale_price', axis=1)
    y_train = train_df['resale_price']
    
    if test_df is not None and 'resale_price' in test_df.columns:
        X_test = test_df.drop('resale_price', axis=1)
        y_test = test_df['resale_price']
        print(f"Using pre-split train ({X_train.shape[0]} samples) and test ({X_test.shape[0]} samples) datasets")
    else:
        # If we don't have separate test data, split training data
        print("No valid test data available, creating a train/test split from training data")
        # Get evaluation settings from config
        test_size = model_config.get('evaluation', {}).get('test_size', 0.2)
        random_state = model_config.get('evaluation', {}).get('random_state', 42)
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=test_size, random_state=random_state
        )
    
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Get model hyperparameters from config
    lr_params = model_config.get('models', {}).get('linear_regression', {})
    lasso_params = model_config.get('models', {}).get('lasso_regression', {})
    ridge_params = model_config.get('models', {}).get('ridge_regression', {})
    
    # Train linear regression model
    if args.linear or args.all:
        print("\nTraining Linear Regression model...")
        lr_model, lr_metrics = train_and_evaluate_model(
            X_train, y_train, X_test, y_test, LinearRegressionModel,
            model_params=lr_params
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
            model_params=lasso_params
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
            model_params=ridge_params
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
