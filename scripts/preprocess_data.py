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
import traceback
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

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
    from src.utils.helpers import load_config
except ImportError as e:
    logger.error(f"Error importing required modules: {e}")
    logger.error("Make sure you're running this script from the project root directory")
    sys.exit(1)

def process_training_data():
    """
    Process training data through preprocessing and feature engineering steps.
    """
    # Load configuration
    model_config = load_config('model_config')
    logger.info("Loaded model configuration")
    
    # Get paths
    data_paths = get_data_paths()
    
    # Set up progress tracking
    preprocessing_steps = ['Loading data', 'Preprocessing', 'Feature engineering', 'Saving results']
    progress_bar = tqdm(preprocessing_steps, desc="Training data processing")
    
    # Load raw training data
    progress_bar.set_description("Loading raw training data")
    logger.info("Loading raw training data...")
    try:
        train_df = load_raw_data(data_paths["train"])
        logger.info(f"Loaded {len(train_df)} records")
        progress_bar.update(1)
    except FileNotFoundError:
        logger.error("Raw training data file not found!")
        logger.error(f"Expected path: {data_paths['train']}")
        progress_bar.close()
        return None
        
    # Process training data
    progress_bar.set_description("Preprocessing data")
    logger.info("Preprocessing data...")
    try:
        # Preprocess data returns a tuple (X, y) for training data
        preprocessed_data = preprocess_data(train_df, is_training=True)
        
        # Check if it's a tuple and handle accordingly
        if isinstance(preprocessed_data, tuple):
            X_preprocessed, y_preprocessed = preprocessed_data
            # Create a DataFrame with both features and target for feature engineering
            preprocessed_df = X_preprocessed.copy()
            preprocessed_df['resale_price'] = y_preprocessed
            logger.info(f"Data preprocessed, {len(preprocessed_df)} records remaining")
        else:
            preprocessed_df = preprocessed_data
            logger.info(f"Data preprocessed, {len(preprocessed_df)} records remaining")
            
        progress_bar.update(1)
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        logger.error(f"Exception details: {str(e)}")
        logger.error(traceback.format_exc())
        progress_bar.close()
        return None
    
    # Engineer features
    progress_bar.set_description("Engineering features")
    logger.info("Engineering features...")
    try:
        # Extract feature configuration
        features_config = model_config.get('features', {})
        feature_selection_method = features_config.get('feature_selection', {}).get('method', 'f_regression')
        feature_selection_k_best = features_config.get('feature_selection', {}).get('k_best', 20)
        
        # Convert k_best to percentile if needed
        if isinstance(feature_selection_k_best, int) and feature_selection_k_best > 0 and feature_selection_k_best <= 100:
            feature_selection_percentile = feature_selection_k_best
        else:
            feature_selection_percentile = 50  # Default to 50 percentile
            
        logger.info(f"Using feature selection: {feature_selection_method}, percentile: {feature_selection_percentile}")
        
        # Get preprocessor with configuration
        preprocessor, numeric_features, categorical_features = engineer_features(
            preprocessed_df, 
            feature_selection_percentile=feature_selection_percentile,
            show_progress=True,
            config=model_config  # Pass the entire config to the function
        )
        
        # Extract X and y for feature transformation
        X = preprocessed_df.drop('resale_price', axis=1, errors='ignore')
        y = preprocessed_df['resale_price'] if 'resale_price' in preprocessed_df.columns else None
        
        # Now fit_transform with the target variable explicitly passed
        feature_matrix = preprocessor.fit_transform(X, y)
        
        # Debug the feature matrix
        logger.info(f"Feature matrix type: {type(feature_matrix)}, shape: {feature_matrix.shape}")
        
        # Convert to DataFrame with appropriate column names
        try:
            feature_names = []
            
            # Check if feature_matrix is actually a sparse matrix and convert it
            if hasattr(feature_matrix, 'toarray'):
                logger.info("Converting sparse matrix to dense array")
                feature_matrix = feature_matrix.toarray()
                logger.info(f"After conversion: shape {feature_matrix.shape}")
            
            # Try to get feature names from the preprocessor directly first
            try:
                if hasattr(preprocessor, 'get_feature_names_out'):
                    feature_names = preprocessor.get_feature_names_out()
                    logger.info(f"Got {len(feature_names)} feature names directly from preprocessor")
            except Exception as e:
                logger.warning(f"Couldn't get feature names directly: {e}")
                
            # Fallback to getting names from each transformer
            if not feature_names:
                for name, transformer, _ in preprocessor.transformers_:
                    if name != 'remainder':  # Skip the 'remainder' transformer if present
                        try:
                            if hasattr(transformer, 'get_feature_names_out'):
                                transformed_features = transformer.get_feature_names_out()
                            elif hasattr(transformer, 'get_feature_names'):  # For older sklearn
                                transformed_features = transformer.get_feature_names()
                            else:
                                logger.warning(f"Transformer {name} has no method to get feature names")
                                continue
                                
                            logger.info(f"Got {len(transformed_features)} features from {name} transformer")
                            feature_names.extend([f"{name}_{feature}" for feature in transformed_features])
                        except AttributeError as e:
                            # Some transformers may not have get_feature_names_out
                            logger.warning(f"Could not get feature names for transformer {name}: {e}")
            
            # Ensure feature_matrix and feature_names match in length - using scalar comparisons
            logger.info(f"Final feature matrix shape: {feature_matrix.shape}")
            logger.info(f"Feature names count: {len(feature_names)}")
            
            # Debugging - Check the types of our values to ensure we're comparing scalars
            feature_cols = int(feature_matrix.shape[1])  # Explicitly cast to int
            feature_names_count = len(feature_names)     # Get the count as a scalar
            
            logger.debug(f"Feature columns: {feature_cols} (type: {type(feature_cols)})")
            logger.debug(f"Feature names count: {feature_names_count} (type: {type(feature_names_count)})")
            
            # Safe comparison with scalars
            if feature_names_count > 0:
                # Another check to avoid array truth value ambiguity
                if feature_cols != feature_names_count:
                    logger.warning(f"Feature matrix columns ({feature_cols}) doesn't match feature names length ({feature_names_count})")
                    logger.warning("Using generic column names instead")
                    feature_names = [f'feature_{i}' for i in range(feature_cols)]
            else:
                logger.info("No feature names were found, using generic names")
                feature_names = [f'feature_{i}' for i in range(feature_cols)]
            
            # Create DataFrame from the transformed features
            logger.info(f"Creating DataFrame with {len(feature_names)} columns")
            try:
                # Check each component individually to avoid ambiguity
                logger.debug(f"Feature matrix type: {type(feature_matrix)}")
                logger.debug(f"Index type: {type(preprocessed_df.index)}, length: {len(preprocessed_df.index)}")
                logger.debug(f"Feature names type: {type(feature_names)}, length: {len(feature_names)}")
                
                # Create the DataFrame with explicit parameter names
                featured_df = pd.DataFrame(
                    data=feature_matrix,
                    index=preprocessed_df.index,
                    columns=feature_names
                )
                logger.info(f"DataFrame created successfully with shape {featured_df.shape}")
            except Exception as specific_e:
                logger.warning(f"Specific DataFrame creation error: {specific_e}")
                # Something might be wrong with feature_names, try again with a simpler approach
                featured_df = pd.DataFrame(
                    data=feature_matrix,
                    index=preprocessed_df.index
                )
                # Add column names after creation
                featured_df.columns = feature_names
                logger.info(f"DataFrame created with post-assignment of columns, shape: {featured_df.shape}")
        except Exception as e:
            logger.warning(f"Error creating DataFrame with named columns: {e}")
            logger.warning(f"Attempting fallback approach with generic column names")
            
            # Ultra fallback - just create DataFrame with generic numbered columns
            try:
                generic_cols = [f'feature_{i}' for i in range(feature_matrix.shape[1])]
                featured_df = pd.DataFrame(
                    feature_matrix, 
                    index=preprocessed_df.index,
                    columns=generic_cols
                )
                logger.info(f"Created DataFrame using fallback with shape {featured_df.shape}")
            except Exception as e2:
                logger.error(f"Even fallback approach failed: {e2}")
                # Last resort - create without column names
                featured_df = pd.DataFrame(feature_matrix)
                featured_df.index = preprocessed_df.index
                logger.warning(f"Created DataFrame without column specifications. Shape: {featured_df.shape}")
        
        logger.info(f"Features engineered, {featured_df.shape[1]} features created")
        progress_bar.update(1)
    except Exception as e:
        logger.error(f"Error during feature engineering: {e}")
        logger.error(f"Exception details: {str(e)}")
        logger.error(traceback.format_exc())
        progress_bar.close()
        return None
    
    # Save processed data
    progress_bar.set_description("Saving processed data")
    logger.info("Saving processed data...")
    try:
        output_path = save_processed_data(featured_df, "train_processed.csv")
        logger.info(f"Processed data saved to: {output_path}")
        progress_bar.update(1)
    except Exception as e:
        logger.error(f"Error saving processed data: {e}")
    
    progress_bar.close()
    return featured_df

def process_test_data():
    """
    Process test data through preprocessing and feature engineering steps.
    """
    # Load configuration
    model_config = load_config('model_config')
    logger.info("Loaded model configuration for test data")
    
    # Get paths
    data_paths = get_data_paths()
    
    # Set up progress tracking
    preprocessing_steps = ['Loading data', 'Preprocessing', 'Feature engineering', 'Saving results']
    progress_bar = tqdm(preprocessing_steps, desc="Test data processing")
    
    # Load raw test data
    progress_bar.set_description("Loading raw test data")
    logger.info("Loading raw test data...")
    try:
        test_df = load_raw_data(data_paths["test"])
        logger.info(f"Loaded {len(test_df)} records")
        progress_bar.update(1)
    except FileNotFoundError:
        logger.error("Raw test data file not found!")
        logger.error(f"Expected path: {data_paths['test']}")
        progress_bar.close()
        return None
    
    # Preprocess the data
    progress_bar.set_description("Preprocessing data")
    logger.info("Preprocessing data...")
    try:
        # For test data, preprocess_data returns a DataFrame directly
        preprocessed_df = preprocess_data(test_df, is_training=False)
        logger.info(f"Data preprocessed, {len(preprocessed_df)} records remaining")
        progress_bar.update(1)
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        progress_bar.close()
        return None
    
    # Engineer features
    progress_bar.set_description("Engineering features")
    logger.info("Engineering features...")
    try:
        # Extract feature configuration
        features_config = model_config.get('features', {})
        feature_selection_method = features_config.get('feature_selection', {}).get('method', 'f_regression')
        feature_selection_k_best = features_config.get('feature_selection', {}).get('k_best', 20)
        
        # Convert k_best to percentile if needed
        if isinstance(feature_selection_k_best, int) and feature_selection_k_best > 0 and feature_selection_k_best <= 100:
            feature_selection_percentile = feature_selection_k_best
        else:
            feature_selection_percentile = 50  # Default to 50 percentile
            
        logger.info(f"Using feature selection: {feature_selection_method}, percentile: {feature_selection_percentile}")
        
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
            
            # Get the preprocessor from the engineering function with configuration
            preprocessor, numeric_features, categorical_features = engineer_features(
                train_preprocessed_df, 
                feature_selection_percentile=feature_selection_percentile,
                show_progress=True,
                config=model_config  # Pass the full configuration
            )
            
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
            
            # Extract scaling method from config
            scaling_method = features_config.get('scaling', 'standard')
            
            # Simplified preprocessor without feature selection
            from sklearn.compose import ColumnTransformer
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
            from sklearn.impute import SimpleImputer
            
            # Choose scaler based on configuration
            if scaling_method == 'minmax':
                scaler = MinMaxScaler()
            elif scaling_method == 'robust':
                scaler = RobustScaler()
            elif scaling_method == 'none':
                scaler = None
            else:  # Default to standard scaling
                scaler = StandardScaler()
            
            # Build transformer pipelines using configuration
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                *([('scaler', scaler)] if scaler else [])
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
            
            # Fit and transform on test data
            feature_matrix = preprocessor.fit_transform(preprocessed_df)
        
        # Debug the feature matrix
        logger.info(f"Test feature matrix type: {type(feature_matrix)}, shape: {feature_matrix.shape}")
        
        # Convert to DataFrame with appropriate column names
        try:
            feature_names = []
            
            # Check if feature_matrix is actually a sparse matrix and convert it
            if hasattr(feature_matrix, 'toarray'):
                logger.info("Converting sparse matrix to dense array")
                feature_matrix = feature_matrix.toarray()
                logger.info(f"After conversion: shape {feature_matrix.shape}")
            
            # Try to get feature names from the preprocessor directly first
            try:
                if hasattr(preprocessor, 'get_feature_names_out'):
                    feature_names = preprocessor.get_feature_names_out()
                    logger.info(f"Got {len(feature_names)} feature names directly from preprocessor")
            except Exception as e:
                logger.warning(f"Couldn't get feature names directly: {e}")
                
            # Fallback to getting names from each transformer
            if not feature_names:
                for name, transformer, _ in preprocessor.transformers_:
                    if name != 'remainder':  # Skip the 'remainder' transformer if present
                        try:
                            if hasattr(transformer, 'get_feature_names_out'):
                                transformed_features = transformer.get_feature_names_out()
                            elif hasattr(transformer, 'get_feature_names'):  # For older sklearn
                                transformed_features = transformer.get_feature_names()
                            else:
                                logger.warning(f"Transformer {name} has no method to get feature names")
                                continue
                                
                            logger.info(f"Got {len(transformed_features)} features from {name} transformer")
                            feature_names.extend([f"{name}_{feature}" for feature in transformed_features])
                        except AttributeError as e:
                            # Some transformers may not have get_feature_names_out
                            logger.warning(f"Could not get feature names for transformer {name}: {e}")
            
            # Ensure feature_matrix and feature_names match in length - using scalar comparisons
            logger.info(f"Final test feature matrix shape: {feature_matrix.shape}")
            logger.info(f"Feature names count: {len(feature_names)}")
            
            # Debugging - Check the types of our values to ensure we're comparing scalars
            feature_cols = int(feature_matrix.shape[1])  # Explicitly cast to int
            feature_names_count = len(feature_names)     # Get the count as a scalar
            
            logger.debug(f"Test feature columns: {feature_cols} (type: {type(feature_cols)})")
            logger.debug(f"Test feature names count: {feature_names_count} (type: {type(feature_names_count)})")
            
            # Safe comparison with scalars
            if feature_names_count > 0:
                # Another check to avoid array truth value ambiguity
                if feature_cols != feature_names_count:
                    logger.warning(f"Test feature matrix columns ({feature_cols}) doesn't match feature names length ({feature_names_count})")
                    logger.warning("Using generic column names instead")
                    feature_names = [f'feature_{i}' for i in range(feature_cols)]
            else:
                logger.info("No feature names were found, using generic names")
                feature_names = [f'feature_{i}' for i in range(feature_cols)]
            
            # Create DataFrame from the transformed features
            logger.info(f"Creating test DataFrame with {len(feature_names)} columns")
            try:
                # Check each component individually to avoid ambiguity
                logger.debug(f"Test feature matrix type: {type(feature_matrix)}")
                logger.debug(f"Test index type: {type(preprocessed_df.index)}, length: {len(preprocessed_df.index)}")
                logger.debug(f"Test feature names type: {type(feature_names)}, length: {len(feature_names)}")
                
                # Create the DataFrame with explicit parameter names
                featured_df = pd.DataFrame(
                    data=feature_matrix,
                    index=preprocessed_df.index,
                    columns=feature_names
                )
                logger.info(f"Test DataFrame created successfully with shape {featured_df.shape}")
            except Exception as specific_e:
                logger.warning(f"Specific test DataFrame creation error: {specific_e}")
                # Something might be wrong with feature_names, try again with a simpler approach
                featured_df = pd.DataFrame(
                    data=feature_matrix,
                    index=preprocessed_df.index
                )
                # Add column names after creation
                featured_df.columns = feature_names
                logger.info(f"Test DataFrame created with post-assignment of columns, shape: {featured_df.shape}")
        except Exception as e:
            logger.warning(f"Error creating DataFrame with named columns: {e}")
            logger.warning(f"Attempting fallback approach with generic column names")
            
            # Ultra fallback - just create DataFrame with generic numbered columns
            try:
                generic_cols = [f'feature_{i}' for i in range(feature_matrix.shape[1])]
                featured_df = pd.DataFrame(
                    feature_matrix, 
                    index=preprocessed_df.index,
                    columns=generic_cols
                )
                logger.info(f"Created test DataFrame using fallback with shape {featured_df.shape}")
            except Exception as e2:
                logger.error(f"Even fallback approach failed: {e2}")
                # Last resort - create without column names
                featured_df = pd.DataFrame(feature_matrix)
                featured_df.index = preprocessed_df.index
                logger.warning(f"Created test DataFrame without column specifications. Shape: {featured_df.shape}")
        
        logger.info(f"Features engineered, {featured_df.shape[1]} features created")
        progress_bar.update(1)
    except Exception as e:
        logger.error(f"Error during feature engineering: {e}")
        logger.error(f"Exception details: {str(e)}")
        logger.error(traceback.format_exc())
        progress_bar.close()
        return None
    
    # Save processed data
    progress_bar.set_description("Saving processed data")
    logger.info("Saving processed data...")
    try:
        output_path = save_processed_data(featured_df, "test_processed.csv")
        logger.info(f"Processed data saved to: {output_path}")
        progress_bar.update(1)
    except Exception as e:
        logger.error(f"Error saving processed data: {e}")
    
    progress_bar.close()
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
    
    # Make sure we're using the parent directory of processed, not the raw file path
    processed_dir = os.path.dirname(data_paths['processed']) if os.path.isfile(data_paths['processed']) else data_paths['processed']
    desc_path = os.path.join(processed_dir, filename)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(desc_path), exist_ok=True)
    
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
            generate_dataset_description(train_df, "train_description.md")
            logger.info("Training data processing complete!")
        else:
            logger.error("Failed to process training data")
    
    # Process test data if requested or if no specific argument given
    if args.test or process_all:
        logger.info("Processing test data...")
        test_df = process_test_data()
        if test_df is not None:
            generate_dataset_description(test_df, "test_description.md")
            logger.info("Test data processing complete!")
        else:
            logger.error("Failed to process test data")
    
    logger.info("Data preprocessing completed")

if __name__ == "__main__":
    main()
