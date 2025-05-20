# """Script for processing raw HDB resale data into model-ready format.

# This script handles the end-to-end data processing workflow for the HDB resale
# price prediction project. It takes raw data from CSV files and transforms it
# through preprocessing and feature engineering to create model-ready datasets.

# The script can process either training data, test data, or both, and saves the 
# processed datasets to the designated processed data directory. The processing
# pipeline includes:

# 1. Loading raw data from the data/raw directory
# 2. Preprocessing steps like cleaning, filtering, and transforming the data
# 3. Feature engineering to create additional predictive features
# 4. Saving the processed data to the data/processed directory

# The processed data is used for model training, evaluation, and making predictions.

# Typical usage:
#     # Process both training and test data
#     $ python scripts/process_data.py
    
#     # Process only training data
#     $ python scripts/process_data.py --train
    
#     # Process only test data
#     $ python scripts/process_data.py --test
# """
# import os
# import sys
# import argparse
# from pathlib import Path

# # Add the project root to the Python path
# script_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(script_dir)
# if project_root not in sys.path:
#     sys.path.append(project_root)

# import pandas as pd

# from src.data.loader import load_raw_data, get_data_paths, save_processed_data
# from src.data.preprocessing import preprocess_data
# from src.data.feature_engineering import engineer_features
# from src.utils.helpers import ensure_dir


# def process_training_data(save=True):
#     """Process training data through preprocessing and feature engineering steps.
    
#     This function implements the complete data processing pipeline for training data:
#     1. Loads raw training data from the data/raw directory
#     2. Applies preprocessing steps to clean and transform the data
#     3. Applies feature engineering to create additional predictive features
#     4. Optionally saves the processed data to the processed data directory
    
#     The function prints progress updates to the console during execution to provide
#     feedback on the processing steps and data dimensions.
    
#     Args:
#         save (bool, optional): Whether to save the processed data to disk.
#             When True, the processed DataFrame is saved to the processed data directory.
#             When False, the DataFrame is only returned without saving.
#             Defaults to True.
        
#     Returns:
#         pd.DataFrame: The fully processed DataFrame with all preprocessing and
#             feature engineering steps applied, ready for model training.
            
#     Raises:
#         FileNotFoundError: If the raw training data file cannot be found.
#         IOError: If the processed data cannot be saved (when save=True).
        
#     Example:
#         >>> # Process training data and save to disk
#         >>> processed_df = process_training_data(save=True)
#         >>> print(f"Processed data has {processed_df.shape[1]} features")
#     """
#     # Get paths
#     data_paths = get_data_paths()
    
#     # Load raw training data
#     print("Loading raw training data...")
#     train_df = load_raw_data(data_paths["train"])
#     print(f"Loaded {len(train_df)} records")
    
#     # Preprocess the data
#     print("Preprocessing data...")
#     preprocessed_df = preprocess_data(train_df, is_training=True)
#     print(f"Data preprocessed, {len(preprocessed_df)} records remaining")
#       # Engineer features
#     print("Engineering features...")
#     # The engineer_features function returns a tuple: (preprocessor, numeric_features, categorical_features)
#     # For the save_processed_data function, we need to transform the data
#     preprocessor, numeric_features, categorical_features = engineer_features(preprocessed_df)
    
#     # Apply the preprocessor to create the actual transformed features
#     # We'll save this transformed DataFrame for model training
#     feature_matrix = preprocessor.fit_transform(preprocessed_df)
    
#     # Convert to DataFrame with appropriate column names
#     # Get the feature names from the column transformer
#     feature_names = []
#     for name, _, _ in preprocessor.transformers_:
#         if name != 'remainder':  # Skip the 'remainder' transformer if present
#             transformed_features = preprocessor.named_transformers_[name].get_feature_names_out()
#             feature_names.extend([f"{name}_{feature}" for feature in transformed_features])
    
#     # Create DataFrame from the transformed features
#     featured_df = pd.DataFrame(
#         feature_matrix, 
#         index=preprocessed_df.index, 
#         columns=feature_names if feature_names else None
#     )
    
#     print(f"Features engineered, {featured_df.shape[1]} features created")
    
#     # Save processed data if requested
#     if save:
#         print("Saving processed data...")
#         save_processed_data(featured_df, "train_processed.csv")
#         print(f"Processed data saved to: {data_paths['processed']}")
    
#     # Return the featured DataFrame and the preprocessor for possible future use
#     return featured_df


# def process_test_data(save=True):
#     """Process test data through preprocessing and feature engineering steps.
    
#     This function implements the complete data processing pipeline for test data:
#     1. Loads raw test data from the data/raw directory
#     2. Applies preprocessing steps to clean and transform the data
#     3. Applies feature engineering to create additional predictive features
#     4. Optionally saves the processed data to the processed data directory
    
#     The function ensures that test data is processed consistently with training data,
#     using the same transformations but with is_training=False to avoid data leakage.
    
#     Args:
#         save (bool, optional): Whether to save the processed data to disk.
#             When True, the processed DataFrame is saved to the processed data directory.
#             When False, the DataFrame is only returned without saving.
#             Defaults to True.
        
#     Returns:
#         pd.DataFrame: The fully processed DataFrame with all preprocessing and
#             feature engineering steps applied, ready for making predictions.
            
#     Raises:
#         FileNotFoundError: If the raw test data file cannot be found.
#         IOError: If the processed data cannot be saved (when save=True).
        
#     Example:
#         >>> # Process test data and save to disk
#         >>> processed_df = process_test_data(save=True)
#         >>> print(f"Processed test data has {processed_df.shape[1]} features")
#     """
#     # Get paths
#     data_paths = get_data_paths()
    
#     # Load raw test data
#     print("Loading raw test data...")
#     test_df = load_raw_data(data_paths["test"])
#     print(f"Loaded {len(test_df)} records")
    
#     # Preprocess the data
#     print("Preprocessing data...")
#     preprocessed_df = preprocess_data(test_df, is_training=False)
#     print(f"Data preprocessed, {len(preprocessed_df)} records remaining")
    
#     # Engineer features
#     print("Engineering features...")
#     featured_df = engineer_features(preprocessed_df)
#     print(f"Features engineered, {featured_df.shape[1]} features created")
    
#     # Save processed data if requested
#     if save:
#         print("Saving processed data...")
#         save_processed_data(featured_df, "test_processed.csv")
#         print(f"Processed data saved to: {os.path.join(data_paths['processed'], 'test_processed.csv')}")
    
#     return featured_df


# def main(args):
#     """Main execution function for the data processing script.
    
#     This function orchestrates the data processing workflow based on command-line arguments:
#     1. Processes training data if --train is specified or no arguments are provided
#     2. Processes test data if --test is specified or no arguments are provided
#     3. Processes both if neither --train nor --test is explicitly specified
    
#     The function serves as the entry point for the script when executed from
#     the command line and delegates to the appropriate data processing functions.
    
#     Args:
#         args (argparse.Namespace): Command-line arguments specifying which data to process.
#             Includes boolean flags:
#             - train: Whether to process training data
#             - test: Whether to process test data
    
#     Returns:
#         None: Results are printed to console and processed data is saved to disk.
        
#     Example:
#         >>> parser = argparse.ArgumentParser()
#         >>> parser.add_argument('--train', action='store_true')
#         >>> args = parser.parse_args(['--train'])
#         >>> main(args)  # Processes only the training data
#     """
#     if args.train:
#         process_training_data()
    
#     if args.test:
#         process_test_data()
    
#     if not args.train and not args.test:
#         print("Processing both training and testing data...")
#         process_training_data()
#         process_test_data()


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Process data for HDB resale price prediction')
#     parser.add_argument('--train', action='store_true', help='Process training data')
#     parser.add_argument('--test', action='store_true', help='Process test data')
    
#     args = parser.parse_args()
    
#     main(args)
