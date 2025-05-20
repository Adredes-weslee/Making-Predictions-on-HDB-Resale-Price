"""
Data Preprocessing Script
========================
This script loads the raw HDB resale transaction data, processes it, and saves the 
processed data for the Streamlit dashboard.

The preprocessing workflow consists of:
1. Loading raw HDB resale transaction data
2. Cleaning and preprocessing the data (handling missing values, date formatting)
3. Engineering features (adding flat age, remaining lease, etc.)
4. Saving the processed dataset for use by the dashboard and models

Usage:
    python preprocess_data.py

The script outputs processed CSV files in the data/processed directory.
"""
import os
import sys
import logging
import argparse
from pathlib import Path
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocessing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import data processing functions
try:
    from src.data.loader import load_raw_data, get_data_paths, save_processed_data
    from src.data.preprocessing import preprocess_data
    from src.data.feature_engineering import engineer_features
except ImportError as e:
    logger.error(f"Error importing required modules: {e}")
    logger.error("Make sure you're running this script from the project root directory")
    sys.exit(1)

def process_training_data():
    """
    Process training data through preprocessing and feature engineering steps.
    """
    # Get paths
    data_paths = get_data_paths()
    
    # Load raw training data
    logger.info("Loading raw training data...")
    try:
        train_df = load_raw_data(data_paths["train"])
        logger.info(f"Loaded {len(train_df)} records")
    except FileNotFoundError:
        logger.error("Raw training data file not found!")
        logger.error(f"Expected path: {data_paths['train']}")
        return None
        
    # Preprocess the data
    logger.info("Preprocessing data...")
    try:
        # Note: preprocess_data returns a tuple (X, y) when is_training=True
        X_train, y_train = preprocess_data(train_df, is_training=True)
        # Add the target back to X for feature engineering
        preprocessed_df = X_train.copy()
        preprocessed_df['resale_price'] = y_train
        logger.info(f"Data preprocessed, {len(preprocessed_df)} records remaining")
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        return None
    
    # Engineer features
    logger.info("Engineering features...")
    try:
        # The engineer_features function returns a tuple: (preprocessor, numeric_features, categorical_features)
        # Since the data is training data, we can fit_transform with the target variable available
        preprocessor, numeric_features, categorical_features = engineer_features(preprocessed_df)
        
        # Check if we have the target column
        has_target = 'resale_price' in preprocessed_df.columns
        
        if not has_target:
            logger.warning("Target column 'resale_price' not found in training data")
              # Apply the preprocessor to create the actual transformed features
        # For training data, we need to extract the target variable before fit_transform
        X = preprocessed_df.drop('resale_price', axis=1, errors='ignore')
        y = preprocessed_df['resale_price'] if 'resale_price' in preprocessed_df.columns else None
        
        # Now fit_transform with the target variable explicitly passed
        feature_matrix = preprocessor.fit_transform(X, y)
        
        # Convert to DataFrame with appropriate column names
        try:
            feature_names = []
            for name, transformer, _ in preprocessor.transformers_:
                if name != 'remainder':  # Skip the 'remainder' transformer if present
                    try:
                        transformed_features = preprocessor.named_transformers_[name].get_feature_names_out()
                        feature_names.extend([f"{name}_{feature}" for feature in transformed_features])
                    except AttributeError:
                        # Some transformers may not have get_feature_names_out
                        logger.warning(f"Could not get feature names for transformer {name}")
            
            # Ensure feature_matrix and feature_names match in length
            if len(feature_names) > 0 and feature_matrix.shape[1] != len(feature_names):
                logger.warning(f"Feature matrix shape {feature_matrix.shape} doesn't match feature names length {len(feature_names)}")
                logger.warning("Using generic column names instead")
                feature_names = [f'feature_{i}' for i in range(feature_matrix.shape[1])]
                
            # Create DataFrame from the transformed features
            featured_df = pd.DataFrame(
                feature_matrix, 
                index=preprocessed_df.index, 
                columns=feature_names if feature_names else [f'feature_{i}' for i in range(feature_matrix.shape[1])]
            )
        except Exception as e:
            logger.warning(f"Error creating DataFrame with named columns: {e}")
            # Fallback to generic column names
            featured_df = pd.DataFrame(
                feature_matrix, 
                index=preprocessed_df.index,
                columns=[f'feature_{i}' for i in range(feature_matrix.shape[1])]
            )
        
        logger.info(f"Features engineered, {featured_df.shape[1]} features created")
    except Exception as e:
        logger.error(f"Error during feature engineering: {e}")
        logger.error(f"Exception details: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None
    
    # Save processed data
    logger.info("Saving processed data...")
    try:
        save_processed_data(featured_df, "train_processed.csv")
        logger.info(f"Processed data saved to: {os.path.join(data_paths['processed'], 'train_processed.csv')}")
    except Exception as e:
        logger.error(f"Error saving processed data: {e}")
    
    return featured_df

def process_test_data():
    """
    Process test data through preprocessing and feature engineering steps.
    """
    # Get paths
    data_paths = get_data_paths()
    
    # Load raw test data
    logger.info("Loading raw test data...")
    try:
        test_df = load_raw_data(data_paths["test"])
        logger.info(f"Loaded {len(test_df)} records")
    except FileNotFoundError:
        logger.error("Raw test data file not found!")
        logger.error(f"Expected path: {data_paths['test']}")
        return None
    
    # Preprocess the data
    logger.info("Preprocessing data...")
    try:
        # For test data, preprocess_data returns a DataFrame directly
        preprocessed_df = preprocess_data(test_df, is_training=False)
        logger.info(f"Data preprocessed, {len(preprocessed_df)} records remaining")
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        return None
    
    # Engineer features
    logger.info("Engineering features...")
    try:
        # For test data, we need a preprocessor that's already been fit on training data
        # Let's check if we have a saved preprocessor, otherwise we need to create one with a special flag for test data
        
        # Try to load the training data to fit the preprocessor first
        try:
            logger.info("Loading training data to fit the preprocessor...")
            train_df = load_raw_data(data_paths["train"])
            # Since preprocess_data returns a tuple for training data, we need to handle it properly
            train_data = preprocess_data(train_df, is_training=True)
            if isinstance(train_data, tuple):
                X_train, y_train = train_data
                # Add the target back to X for feature engineering
                train_preprocessed_df = X_train.copy()
                train_preprocessed_df['resale_price'] = y_train
            else:
                train_preprocessed_df = train_data
              # Get the preprocessor from the engineering function (it won't be fitted yet)
            preprocessor, numeric_features, categorical_features = engineer_features(train_preprocessed_df)
            
            # Fit the preprocessor on training data - with target variable
            logger.info("Fitting the preprocessor on training data...")
            X_train = train_preprocessed_df.drop('resale_price', axis=1, errors='ignore')
            y_train = train_preprocessed_df['resale_price'] if 'resale_price' in train_preprocessed_df.columns else None
            
            if y_train is None:
                logger.warning("No target variable found in training data for fitting!")
                raise ValueError("Target variable required for feature selection")
                
            preprocessor.fit(X_train, y_train)
            
            # Now transform the test data (don't fit again)
            logger.info("Transforming test data using the fitted preprocessor...")
            feature_matrix = preprocessor.transform(preprocessed_df)
            
        except Exception as train_error:
            # If we can't load training data, create a simplified preprocessor without feature selection
            logger.warning(f"Could not load training data to fit preprocessor: {train_error}")
            logger.warning("Creating simplified preprocessor for test data without feature selection...")
            
            # For test data only, we'll create a preprocessor without feature selection
            # since feature selection requires target values
            numeric_features = [col for col in preprocessed_df.columns if preprocessed_df[col].dtype in ['float64', 'int64']]
            categorical_features = [col for col in preprocessed_df.columns if preprocessed_df[col].dtype == 'object']
            
            # Simplified preprocessor without feature selection
            from sklearn.compose import ColumnTransformer
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler, OneHotEncoder
            from sklearn.impute import SimpleImputer
            
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])
            
            categorical_transformer = Pipeline(steps=[
                ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
            ])
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)
                ],
                remainder='drop'
            )
            
            # Fit and transform in one step on the test data
            logger.info("Fit-transforming test data with simplified preprocessor...")
            feature_matrix = preprocessor.fit_transform(preprocessed_df)
        
        # Convert to DataFrame with appropriate column names
        try:
            feature_names = []
            for name, transformer, _ in preprocessor.transformers_:
                if name != 'remainder':  # Skip the 'remainder' transformer if present
                    try:
                        transformed_features = preprocessor.named_transformers_[name].get_feature_names_out()
                        feature_names.extend([f"{name}_{feature}" for feature in transformed_features])
                    except AttributeError:
                        # Some transformers may not have get_feature_names_out
                        logger.warning(f"Could not get feature names for transformer {name}")
            
            # Ensure feature_matrix and feature_names match in length
            if len(feature_names) > 0 and feature_matrix.shape[1] != len(feature_names):
                logger.warning(f"Feature matrix shape {feature_matrix.shape} doesn't match feature names length {len(feature_names)}")
                logger.warning("Using generic column names instead")
                feature_names = [f'feature_{i}' for i in range(feature_matrix.shape[1])]
                
            # Create DataFrame from the transformed features
            featured_df = pd.DataFrame(
                feature_matrix, 
                index=preprocessed_df.index, 
                columns=feature_names if feature_names else [f'feature_{i}' for i in range(feature_matrix.shape[1])]
            )
        except Exception as e:
            logger.warning(f"Error creating DataFrame with named columns: {e}")
            # Fallback to generic column names
            featured_df = pd.DataFrame(
                feature_matrix, 
                index=preprocessed_df.index,
                columns=[f'feature_{i}' for i in range(feature_matrix.shape[1])]
            )
        
        logger.info(f"Features engineered, {featured_df.shape[1]} features created")
    except Exception as e:
        logger.error(f"Error during feature engineering: {e}")
        logger.error(f"Exception details: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None
    
    # Save processed data
    logger.info("Saving processed data...")
    try:
        save_processed_data(featured_df, "test_processed.csv")
        logger.info(f"Processed data saved to: {os.path.join(data_paths['processed'], 'test_processed.csv')}")
    except Exception as e:
        logger.error(f"Error saving processed data: {e}")
    
    return featured_df

def generate_dataset_description(df, filename):
    """
    Generate and save a description of the dataset
    """
    if df is None:
        return
        
    # Generate description statistics and metadata
    description = f"""
# HDB Resale Price Dataset Description

## Overview
- Total records: {df.shape[0]}
- Number of features: {df.shape[1]}

## Features
{pd.DataFrame({'Feature': df.columns, 'Non-Null Count': df.count().values, 
               'Data Type': df.dtypes.values}).to_string(index=False)}

## Summary Statistics
{df.describe().round(2).to_string()}
    """
    
    # Save description
    data_paths = get_data_paths()
    desc_path = os.path.join(data_paths['processed'], filename)
    with open(desc_path, 'w') as f:
        f.write(description)
    logger.info(f"Dataset description saved to {desc_path}")

def main():
    """
    Main execution function for data preprocessing
    """
    parser = argparse.ArgumentParser(description='Process data for HDB resale price prediction')
    parser.add_argument('--train', action='store_true', help='Process training data')
    parser.add_argument('--test', action='store_true', help='Process test data')
    
    args = parser.parse_args()
    
    # If no args specified, process both
    process_all = not args.train and not args.test
    
    logger.info("Starting data preprocessing")
    
    # Process training data if requested or if no specific argument given
    if args.train or process_all:
        logger.info("Processing training data...")
        train_df = process_training_data()
        if train_df is not None:
            generate_dataset_description(train_df, "train_data_description.md")
    
    # Process test data if requested or if no specific argument given
    if args.test or process_all:
        logger.info("Processing test data...")
        test_df = process_test_data()
        if test_df is not None:
            generate_dataset_description(test_df, "test_data_description.md")
    
    logger.info("Data preprocessing completed")

if __name__ == "__main__":
    main()
