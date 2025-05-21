"""
Data Preprocessing Script (FOR EXPLORATORY PURPOSES)
=================================================
This script loads the raw HDB resale transaction data, processes it, and saves the 
processed data primarily for exploratory data analysis and visualization in the
Streamlit dashboard.

⚠️ IMPORTANT: This script uses a different preprocessing approach than train_pipeline_model.py ⚠️
Data processed with this script may NOT be fully compatible with pipeline models.
For production model training, please use train_pipeline_model.py which implements
a consistent scikit-learn pipeline approach.

The preprocessing workflow consists of:
1. Loading raw HDB resale transaction data
2. Cleaning and preprocessing the data (handling missing values, date formatting)
3. Engineering features (adding flat age, remaining lease, etc.)
4. Saving the processed dataset for exploratory analysis

Usage:
    python preprocess_data.py [--no-save] [--debug]

Options:
    --no-save: Run preprocessing without saving the outputs
    --debug: Enable additional debug logging
"""
import os
import sys
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Add the project root to the path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

# Import project modules
from src.data.loader import get_data_paths, load_raw_data, save_processed_data
from src.data.preprocessing import clean_data
from src.utils.helpers import get_project_root, load_config, setup_logging

# Setup logging
logger = logging.getLogger(__name__)

def setup_directories():
    """Create necessary directories if they don't exist."""
    base_dir = get_project_root()
    processed_dir = base_dir / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    return processed_dir

def extract_features_info(df):
    """
    Extract and save feature information to aid in interpretation and debugging.
    
    Args:
        df: DataFrame with features
    
    Returns:
        Dictionary with feature metadata
    """
    # Get numerical and categorical features
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Feature info
    feature_info = {
        'numerical_features': numerical_features,
        'categorical_features': categorical_features,
        'total_features': len(df.columns),
        'numerical_count': len(numerical_features),
        'categorical_count': len(categorical_features),
        'feature_names': df.columns.tolist(),
        'timestamp': datetime.now().isoformat(),
        'preprocessing_method': 'exploratory_script',  # Mark the preprocessing method
        'warning': 'This data was processed with the exploratory script and may not be compatible with pipeline models.'
    }
    
    return feature_info

def engineer_temporal_features(df):
    """
    Add temporal features based on transaction date.
    
    Note: These features are created for exploratory analysis and may differ
    from those created by the training pipeline.
    
    Args:
        df: DataFrame with transaction date column
        
    Returns:
        DataFrame with additional temporal features
    """
    logger.info("Engineering temporal features (exploratory version)...")
    
    # Check if date-related columns exist
    date_columns = [col for col in df.columns if any(term in col.lower() 
                                                    for term in ['year', 'month', 'date'])]
    
    if 'month' in df.columns and 'year' in df.columns:
        # Create transaction date if month and year are available
        try:
            df['transaction_date'] = pd.to_datetime(
                df['year'].astype(str) + '-' + df['month'].astype(str) + '-01'
            )
            logger.debug("Created transaction_date from year and month columns")
        except Exception as e:
            logger.warning(f"Could not create transaction_date: {e}")
    
    # Try to parse Tranc_YearMonth if available
    if 'Tranc_YearMonth' in df.columns:
        try:
            df['transaction_date'] = pd.to_datetime(df['Tranc_YearMonth'])
            
            # Extract year, month from the date
            df['transaction_year'] = df['transaction_date'].dt.year
            df['transaction_month'] = df['transaction_date'].dt.month
            df['transaction_quarter'] = df['transaction_date'].dt.quarter
            
            logger.debug("Created temporal features from Tranc_YearMonth")
        except Exception as e:
            logger.warning(f"Could not process Tranc_YearMonth: {e}")
    
    return df

def engineer_property_features(df):
    """
    Engineer property-related features like age, remaining lease, etc.
    
    Note: These features are created for exploratory analysis and may differ
    from those created by the training pipeline.
    
    Args:
        df: DataFrame with property attributes
        
    Returns:
        DataFrame with additional property features
    """
    logger.info("Engineering property features (exploratory version)...")
    current_year = datetime.now().year
    
    # Calculate flat age if lease commence date is available
    if 'lease_commence_date' in df.columns:
        try:
            # Calculate flat age at time of transaction
            if 'transaction_year' in df.columns:
                df['flat_age'] = df['transaction_year'] - df['lease_commence_date']
            else:
                df['flat_age'] = current_year - df['lease_commence_date']
                
            logger.debug("Added flat_age feature")
        except Exception as e:
            logger.warning(f"Could not calculate flat_age: {e}")
    
    # Parse remaining lease if available
    if 'remaining_lease' in df.columns:
        try:
            # Extract years and months from the 'remaining_lease' string
            # Pattern typically: "XX years YY months"
            df['remaining_lease_years'] = df['remaining_lease'].str.extract(r'(\d+) years')
            df['remaining_lease_years'] = pd.to_numeric(df['remaining_lease_years'][0], errors='coerce')
            
            df['remaining_lease_months'] = df['remaining_lease'].str.extract(r'(\d+) months')
            df['remaining_lease_months'] = pd.to_numeric(df['remaining_lease_months'][0], errors='coerce')
            
            # Calculate total remaining lease in months
            df['remaining_lease_total_months'] = (df['remaining_lease_years'] * 12 + 
                                                df['remaining_lease_months'])
                                                
            logger.debug("Parsed remaining_lease into years and months")
        except Exception as e:
            logger.warning(f"Could not parse remaining_lease: {e}")
    
    # Process floor area if available
    if 'floor_area_sqm' in df.columns:
        try:
            # Ensure floor area is numeric
            df['floor_area_sqm'] = pd.to_numeric(df['floor_area_sqm'], errors='coerce')
            
            # Add floor area categories
            bins = [0, 50, 75, 100, 125, 150, 200, np.inf]
            labels = ['Tiny (≤50m²)', 'Small (51-75m²)', 'Medium (76-100m²)', 
                     'Large (101-125m²)', 'XLarge (126-150m²)', 'Huge (151-200m²)', 'Mansion (>200m²)']
            df['floor_area_category'] = pd.cut(df['floor_area_sqm'], bins=bins, labels=labels)
            
            logger.debug("Added floor_area_category feature")
        except Exception as e:
            logger.warning(f"Could not create floor area categories: {e}")
    
    return df

def preprocess_dataset(df, is_training=True):
    """
    Preprocess a dataset with cleaning and feature engineering.
    
    This consolidated function handles both train and test data.
    
    ⚠️ Note: This preprocessing is for exploratory purposes and
    may differ from the preprocessing pipeline used in model training.
    
    Args:
        df: Raw DataFrame to preprocess
        is_training: Whether this is training data (includes target)
    
    Returns:
        Preprocessed DataFrame
    """
    logger.info(f"Starting preprocessing for {'training' if is_training else 'testing'} data")
    logger.info(f"Raw data shape: {df.shape}")
    
    # Clean the data first
    df = clean_data(df)
    logger.info(f"Data shape after cleaning: {df.shape}")
    
    # Engineer features
    df = engineer_temporal_features(df)
    df = engineer_property_features(df)
    logger.info(f"Data shape after feature engineering: {df.shape}")
    
    # Extract feature information
    feature_info = extract_features_info(df)
    logger.info(f"Processed features: {feature_info['total_features']} total, " 
               f"{feature_info['numerical_count']} numerical, "
               f"{feature_info['categorical_count']} categorical")
    
    return df, feature_info

def main(args):
    """Main function to execute the preprocessing workflow."""
    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(log_level)
    
    logger.info("=" * 80)
    logger.warning("⚠️ THIS SCRIPT IS PRIMARILY FOR EXPLORATORY DATA ANALYSIS ⚠️")
    logger.warning("Data processed with this script may not be compatible with pipeline models.")
    logger.warning("For model training, consider using train_pipeline_model.py instead.")
    logger.info("=" * 80)
    
    logger.info("Starting HDB resale data preprocessing (exploratory version)")
    
    try:
        # Get data paths
        data_paths = get_data_paths()
        processed_dir = setup_directories()
        
        # Load training data
        train_path = data_paths["train"]
        logger.info(f"Loading training data from {train_path}")
        
        try:
            train_df = load_raw_data(train_path)
            logger.info(f"Loaded training data with shape: {train_df.shape}")
        except FileNotFoundError:
            logger.error(f"Training data file not found at {train_path}")
            return
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            return
        
        # Preprocess training data
        processed_train_df, train_features_info = preprocess_dataset(train_df, is_training=True)
        logger.info(f"Training data preprocessing complete. Final shape: {processed_train_df.shape}")
        
        # Save feature info for debugging and documentation
        if not args.no_save:
            # Add compatibility warning to filename
            feature_info_path = processed_dir / "feature_info_exploratory.json"
            with open(feature_info_path, 'w', encoding='utf-8') as f:
                json.dump(train_features_info, f, indent=2)
            logger.info(f"Feature information saved to {feature_info_path}")
            
            # Create a README file explaining the difference
            readme_path = processed_dir / "README.md"
            if not readme_path.exists():
                with open(readme_path, 'w', encoding='utf-8') as f:
                    f.write("""# Processed Data Directory

This directory contains processed data files created by different preprocessing scripts:

## Exploratory Files
Files with `_exploratory` in the name are created by the `preprocess_data.py` script and are 
primarily intended for exploratory data analysis and visualization in the Streamlit dashboard.

⚠️ **Warning**: These files may not be compatible with models trained using the pipeline approach.

## Pipeline-Compatible Files
Files without the `_exploratory` suffix are created by the `train_pipeline_model.py` script and
use a consistent scikit-learn pipeline approach that ensures compatibility between training and prediction.

For model training and evaluation, please use the pipeline-compatible files.
""")
        
        # Check if test data exists and process it
        test_path = data_paths["test"]
        if os.path.exists(test_path):
            logger.info(f"Loading test data from {test_path}")
            
            try:
                test_df = load_raw_data(test_path)
                logger.info(f"Loaded test data with shape: {test_df.shape}")
                
                # Preprocess test data with same steps as training data
                processed_test_df, _ = preprocess_dataset(test_df, is_training=False)
                logger.info(f"Test data preprocessing complete. Final shape: {processed_test_df.shape}")
                
                # Save processed test data
                if not args.no_save:
                    test_output_path = processed_dir / "test_processed_exploratory.csv"
                    processed_test_df.to_csv(test_output_path, index=False)
                    logger.info(f"Processed test data saved to {test_output_path}")
            except Exception as e:
                logger.error(f"Error processing test data: {e}")
        else:
            logger.warning(f"Test data file not found at {test_path}. Skipping test processing.")
            
            # Create a test set from training data if test file doesn't exist
            logger.info("Creating test set from training data using train-test split")
            split_size = 0.2  # Use 20% for test
            
            try:
                if 'resale_price' in processed_train_df.columns:
                    # Split the data with stratification if possible
                    X = processed_train_df.drop(columns=['resale_price'])
                    y = processed_train_df['resale_price']
                    
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=split_size, random_state=42
                    )
                    
                    # Reconstruct complete dataframes
                    processed_train_split = pd.concat([X_train, y_train], axis=1)
                    processed_test_df = pd.concat([X_test, y_test], axis=1)
                    
                    # Replace the original train DataFrame with the split version
                    processed_train_df = processed_train_split
                    
                    logger.info(f"Created test set with {len(processed_test_df)} samples")
                    
                    # Save the split test data
                    if not args.no_save:
                        test_output_path = processed_dir / "test_split_exploratory.csv"
                        processed_test_df.to_csv(test_output_path, index=False)
                        logger.info(f"Split test data saved to {test_output_path}")
                else:
                    logger.warning("No 'resale_price' column found. Cannot split data.")
            except Exception as e:
                logger.error(f"Error splitting data: {e}")
        
        # Save processed training data
        if not args.no_save:
            train_output_path = processed_dir / "train_processed_exploratory.csv" 
            processed_train_df.to_csv(train_output_path, index=False)
            logger.info(f"Processed training data saved to {train_output_path}")
        
        logger.info("Preprocessing completed successfully!")
        logger.warning("Remember: This data is processed for exploratory purposes.")
        logger.warning("For model training, use train_pipeline_model.py instead.")
        
    except Exception as e:
        logger.exception(f"An unexpected error occurred during preprocessing: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess HDB resale data for exploratory analysis")
    parser.add_argument("--no-save", action="store_true", help="Run without saving outputs")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    main(args)