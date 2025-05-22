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
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from src.data.preprocessing_pipeline import create_standard_preprocessing_pipeline
from src.data.feature_engineering import get_numeric_features, get_categorical_features
from datetime import datetime

# Define model options
MODEL_TYPES = {
    'linear': LinearRegression,
    'ridge': Ridge,
    'lasso': Lasso
}

def load_training_data(data_path):
    """Load and prepare training data with consistent preprocessing.
    
    Args:
        data_path: Path to the training data CSV file
        
    Returns:
        X: Features DataFrame
        y: Target Series
    """
    try:
        from src.data.preprocessing_pipeline import prepare_data_for_modeling
        
        # Load the data
        df = pd.read_csv(data_path, encoding='utf-8')
        print(f"Loaded data columns: {df.columns[:10]}...")
        
        # Use the prepare_data_for_modeling function
        X, y, numerical_features, categorical_features = prepare_data_for_modeling(
            df, is_training=True, drop_high_missing=True)
        
        print(f"Preprocessed data: {X.shape[0]} samples with {X.shape[1]} features")
        print(f"  - {len(numerical_features)} numerical features")
        print(f"  - {len(categorical_features)} categorical features")
        
        # Store the feature lists as attributes on X for easier access
        X._numerical_features = numerical_features
        X._categorical_features = categorical_features
        
        return X, y
    except Exception as e:
        print(f"Error loading training data: {str(e)}")
        raise


def create_preprocessing_pipeline(X, categorical_features=None, numerical_features=None, feature_percentile=50):
    """Create a preprocessing pipeline for the given data.
    
    This function builds a scikit-learn preprocessing pipeline that handles both categorical
    and numerical features, with appropriate transformations for each.
    
    Args:
        X (pd.DataFrame): Input features dataframe
        categorical_features (list, optional): List of categorical feature names
        numerical_features (list, optional): List of numerical feature names  
        feature_percentile (int, optional): Percentile for feature selection
        
    Returns:
        tuple: (preprocessor, categorical_features, numerical_features)
    """
    # Use features identified during data loading if not explicitly provided
    if categorical_features is None and hasattr(X, '_categorical_features'):
        categorical_features = X._categorical_features
    elif categorical_features is None:
        categorical_features = get_categorical_features(X)
        
    if numerical_features is None and hasattr(X, '_numerical_features'):
        numerical_features = X._numerical_features
    elif numerical_features is None:
        numerical_features = get_numeric_features(X)
    
    # Use the preprocessing pipeline from src.data.preprocessing_pipeline
    preprocessor = create_standard_preprocessing_pipeline(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        feature_percentile=feature_percentile
    )
    
    return preprocessor, categorical_features, numerical_features


def train_and_save_pipeline_model(model_type='linear', data_path=None, model_name=None, 
                                 model_dir=None, test_size=0.2, random_state=42, 
                                 feature_percentile=50):
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
    
    # Save processed data for consistency across training and prediction
    from src.data.preprocessing_pipeline import save_processed_data
    root_dir = Path(__file__).parent.parent.parent
    processed_data_path = save_processed_data(X, y, 
        output_path=os.path.join(root_dir, "data", "processed", "train_pipeline_processed.csv"))
    print(f"Saved processed training data to {processed_data_path}")
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    
    # Create preprocessing pipeline
    preprocessor, categorical_features, numerical_features = create_preprocessing_pipeline(
        X, 
        feature_percentile=feature_percentile  # Pass the parameter here
    )
        
    # Add precise feature count debugging
    print("\n=== EXACT FEATURE COUNT DEBUG ===")
    print(f"Original data shape: {X.shape[1]} columns")
    print(f"  - Categorical features: {len(categorical_features)}")
    print(f"  - Numerical features: {len(numerical_features)}")
    
    # FIXED VERSION: Pass both X and y to the fit method
    print("Fitting preprocessor on small sample to analyze feature counts...")
    
    try:
        # First, examine just the encoding step before feature selection
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        
        # Create a simple transformer without feature selection
        simple_transformer = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numerical_features),
                ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features),
            ],
            remainder='drop',
            verbose_feature_names_out=False
        )
        
        # Fit and transform a small sample
        simple_transformer.fit(X_train.iloc[:])
        X_encoded = simple_transformer.transform(X_train.iloc[:1])
        
        print(f"After encoding (before selection): {X_encoded.shape[1]} features")
        
        # Now fit the full preprocessor with feature selection
        # Important: pass both X and y to the fit method
        preprocessor_clone = clone(preprocessor)
        preprocessor_clone.fit(X_train.iloc[:100], y_train.iloc[:100])
        X_transformed = preprocessor_clone.transform(X_train.iloc[:1])
        
        print(f"After feature selection: {X_transformed.shape[1]} features")
        print(f"Feature selection kept {X_transformed.shape[1]/X_encoded.shape[1]:.1%} of features")
        
        # Try to get feature names if available
        feature_names = preprocessor_clone.get_feature_names_out()
        print(f"  - Feature names count: {len(feature_names)}")
        
        # Count features by type
        cat_features = [f for f in feature_names if f.startswith('cat__')]
        num_features = [f for f in feature_names if f.startswith('num__')]
        print(f"  - Categorical features after selection: {len(cat_features)}")
        print(f"  - Numerical features after selection: {len(num_features)}")
        
        print(f"Feature selection kept: {feature_percentile}% (expected to keep {int(X_encoded.shape[1] * feature_percentile/100)} features)")
    except Exception as e:
        print(f"  - Debug error: {str(e)}")
        print(f"  - Debug error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
    
    print("================================\n")
    
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
    
    # Add this after your feature_info JSON is saved
    metrics_path = os.path.join(model_dir, f"{model_name}_metrics.json")
    with open(metrics_path, 'w') as f:
        # Add a timestamp to the metrics
        metrics_with_timestamp = {
            **metrics,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_type': model_type,
            'feature_percentile': feature_percentile
        }
        json.dump(metrics_with_timestamp, f, indent=2)
    print(f"Model metrics saved to {metrics_path}")

    # Update the return value
    return {
        'metrics': metrics,
        'pipeline_path': pipeline_path,
        'feature_info_path': feature_path,
        'metrics_path': metrics_path,  # Add this line
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
