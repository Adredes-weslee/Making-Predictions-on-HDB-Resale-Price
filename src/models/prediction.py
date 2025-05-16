"""Module for making predictions with trained HDB price prediction models.

This module provides functions for loading trained machine learning models and using
them to make predictions on new HDB property data. It handles the entire prediction
workflow including model loading, data preprocessing, generating predictions, and
creating submission files or batch predictions across multiple models.

The module supports different prediction scenarios:
1. Single model prediction - Making predictions with a single trained model
2. Batch prediction - Making predictions with multiple models for comparison
3. Submission generation - Creating formatted output files for submissions

The functions handle both raw and preprocessed data, taking care of any necessary
preprocessing steps before prediction when required.

Typical usage:
    >>> import pandas as pd
    >>> from src.models.prediction import load_model, predict
    >>> 
    >>> # Load a trained model
    >>> model = load_model('models/pipe_lr_ss.pkl')
    >>> 
    >>> # Load new data for prediction
    >>> test_data = pd.read_csv('data/processed/test_data.csv')
    >>> 
    >>> # Make predictions
    >>> predictions = predict(model, test_data)
"""
from typing import Dict, List, Optional, Tuple, Union
import pickle
import numpy as np
import pandas as pd

from src.data.preprocessing import preprocess_data
from src.models.base import Model


def load_model(model_path: str) -> object:
    """Load a trained machine learning model from a pickle file.
    
    This function deserializes and loads a previously trained and saved model from
    a pickle file. The model can be any machine learning model that was serialized
    using Python's pickle module, typically sklearn pipelines or custom models
    implementing the Model interface.
    
    Args:
        model_path (str): The file path to the saved model pickle file. Should be
            an absolute or relative path pointing to a .pkl or .pickle file.
        
    Returns:
        object: The loaded model object that can be used for making predictions.
        
    Raises:
        FileNotFoundError: If the specified model file doesn't exist.
        pickle.UnpicklingError: If the file exists but cannot be unpickled properly.
        
    Example:
        >>> from src.models.prediction import load_model
        >>> # Load a trained linear regression model
        >>> model = load_model('models/pipe_lr_ss.pkl')
        >>> # The model is now ready to use for predictions
        >>> predictions = model.predict(X_test)
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


def predict(
    model: object,
    X: pd.DataFrame,
    preprocessed: bool = False
) -> np.ndarray:
    """Make predictions using a trained model with optional preprocessing.
    
    This function generates predictions for new data using a trained model. If the
    input data is not already preprocessed, it automatically applies the necessary
    preprocessing steps before prediction. This ensures that the data format matches
    what the model expects.
    
    Args:
        model (object): A trained model object with a predict method, typically
            an sklearn model, pipeline, or custom model implementing the Model interface.
        X (pd.DataFrame): DataFrame containing the features for which predictions
            are to be generated. The columns should match those used during model training.
        preprocessed (bool, optional): Flag indicating whether the input data is
            already preprocessed. If False, preprocessing will be applied before prediction.
            Defaults to False.
        
    Returns:
        np.ndarray: An array of predictions. The exact format depends on the model,
            but typically contains predicted property prices for regression models.
        
    Raises:
        ValueError: If the model doesn't have a predict method or if the input
            data doesn't have the expected features.
          Example:
        >>> import pandas as pd
        >>> from src.models.prediction import load_model, predict
        >>> from src.data.preprocessing import preprocess_data
        >>> 
        >>> # Load model and test data
        >>> model = load_model('models/pipe_lr_ss.pkl')
        >>> test_data = pd.read_csv('data/processed/test_data.csv')
        >>> 
        >>> # Make predictions on raw data (will be preprocessed)
        >>> raw_predictions = predict(model, test_data, preprocessed=False)
        >>> 
        >>> # If data is already preprocessed
        >>> preprocessed_data = preprocess_data(test_data, is_training=False)
        >>> predictions = predict(model, preprocessed_data, preprocessed=True)
    """
    if not preprocessed:
        X = preprocess_data(X, is_training=False)
        
    return model.predict(X)


def create_submission(
    model: object,
    X: pd.DataFrame,
    preprocessed: bool = False,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """Create a formatted submission file with predictions for external platforms.
      This function generates predictions for new data and formats them into a submission
    dataframe suitable for platforms like Kaggle. The function can optionally save
    the submission to a CSV file. If the input data contains an 'id' column, it will
    be included in the submission as 'Id' (capitalized).
    
    Args:
        model (object): A trained model object with a predict method, typically
            an sklearn model, pipeline, or custom model implementing the Model interface.
        X (pd.DataFrame): DataFrame containing the features for which predictions
            are to be generated. If it contains an 'id' column, this will be used
            in the submission file.
        preprocessed (bool, optional): Flag indicating whether the input data is
            already preprocessed. If False, preprocessing will be applied before prediction.
            Defaults to False.
        output_path (str, optional): File path where the submission CSV should be saved.
            If None, the submission will only be returned as a DataFrame without saving.
            Defaults to None.
        
    Returns:
        pd.DataFrame: A DataFrame containing the submission data, typically with
            'Id' and 'Predicted' columns, ready for submission to platforms like Kaggle.
        
    Raises:
        ValueError: If the model doesn't have a predict method or if the input
            data doesn't have the expected features.
        
    Example:
        >>> import pandas as pd
        >>> from src.models.prediction import load_model, create_submission
        >>> 
        >>> # Load model and test data
        >>> model = load_model('models/pipe_lr_ss.pkl')
        >>> test_data = pd.read_csv('data/raw/test.csv')
        >>> 
        >>> # Create submission with predictions
        >>> submission = create_submission(
        ...     model,
        ...     test_data,
        ...     preprocessed=False,
        ...     output_path='submissions/lr_submission.csv'
        ... )
        >>> print(f"Generated {len(submission)} predictions")
    """
    # Store the ID column
    id_col = X['id'] if 'id' in X.columns else None
    
    # Make predictions
    if not preprocessed:
        X_processed = preprocess_data(X, is_training=False)
        predictions = model.predict(X_processed)
    else:
        predictions = model.predict(X)
    
    # Create submission dataframe
    if id_col is not None:
        submission = pd.DataFrame({
            'Id': id_col,
            'Predicted': predictions
        })
    else:
        submission = pd.DataFrame({
            'Predicted': predictions
        })
    
    # Save submission file
    if output_path:
        submission.to_csv(output_path, index=False)
    
    return submission


def batch_predict(
    models: Dict[str, object],
    X: pd.DataFrame,
    preprocessed: bool = False
) -> pd.DataFrame:
    """Make predictions with multiple models for comparison.
    
    This function generates predictions using multiple models for the same input data,
    allowing for side-by-side comparison of different model outputs. It handles
    preprocessing if needed and organizes the results in a DataFrame for easy analysis.
    
    Args:
        models (Dict[str, object]): Dictionary mapping model names to trained model
            objects. Each model should have a predict method.
        X (pd.DataFrame): DataFrame containing the features for which predictions
            are to be generated.
        preprocessed (bool, optional): Flag indicating whether the input data is
            already preprocessed. If False, preprocessing will be applied before prediction.
            Defaults to False.
        
    Returns:
        pd.DataFrame: A DataFrame containing predictions from all models, with columns
            named according to the keys in the models dictionary.
        
    Example:
        >>> # Load multiple models
        >>> models = {
        ...     'linear': load_model('models/pipe_lr_ss.pkl'),
        ...     'ridge': load_model('models/pipe_rr_ss.pkl'),
        ...     'lasso': load_model('models/pipe_lasso_ss.pkl')
        ... }
        >>> 
        >>> # Make predictions with all models
        >>> test_data = pd.read_csv('data/processed/test_data.csv')
        >>> predictions_df = batch_predict(models, test_data, preprocessed=True)
        >>> 
        >>> # Compare first few predictions
        >>> print(predictions_df.head())
    """
    # Preprocess data if needed
    if not preprocessed:
        X_processed = preprocess_data(X, is_training=False)
    else:
        X_processed = X
    
    # Create dictionary to store predictions
    predictions = {}
    
    # Get predictions from each model
    for name, model in models.items():
        predictions[name] = model.predict(X_processed)
    
    # Combine into DataFrame
    return pd.DataFrame(predictions)
