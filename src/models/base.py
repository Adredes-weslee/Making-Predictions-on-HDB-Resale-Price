"""Base model class for HDB resale price prediction models.

This module defines the abstract base class for all HDB resale price prediction models,
establishing a common interface and shared functionality. It provides a foundation
for implementing different regression algorithms while maintaining consistent behavior
for training, evaluation, prediction, and model persistence.

The Model abstract base class defines the core lifecycle of prediction models:
1. Model initialization with configurable parameters
2. Feature engineering and pipeline creation
3. Model training on housing data
4. Making predictions on new data
5. Model evaluation using standard metrics
6. Saving and loading trained models

By inheriting from this base class, model implementations can focus on their specific
algorithm details while leveraging common infrastructure for preprocessing, evaluation,
and persistence.

Typical usage:
    >>> import pandas as pd
    >>> from src.models.linear import LinearRegressionModel
    >>> 
    >>> # Load training data
    >>> X_train = pd.read_csv('data/processed/X_train.csv')
    >>> y_train = pd.read_csv('data/processed/y_train.csv')['resale_price']
    >>> 
    >>> # Initialize and train model
    >>> model = LinearRegressionModel()
    >>> model.fit(X_train, y_train)
    >>> 
    >>> # Save trained model
    >>> model.save('linear_model.pkl')
"""
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.data.feature_engineering import engineer_features


class Model:
    """Abstract base class for HDB resale price prediction models.
    
    This class serves as a template and common interface for all model implementations
    in the HDB resale price prediction project. It defines the standard lifecycle of
    a model including initialization, training, prediction, evaluation and persistence.
    
    The class is designed to work with scikit-learn style estimators and pipelines,
    providing a consistent interface regardless of the underlying algorithm. It also
    incorporates feature engineering as part of the model pipeline to ensure consistent
    data transformations during both training and prediction.
    
    Attributes:
        model_type (str): Identifier for the type of model (e.g., "lr", "ridge", "lasso").
        feature_selection_percentile (int): Percentage of features to select based on
            mutual information scores with the target variable.
        random_state (int): Random seed for reproducibility across runs.
        model_dir (str): Directory path for saving and loading model files.
        model (object): The trained model object, None before training.
        preprocessor (ColumnTransformer): Feature transformation pipeline.
        pipeline (Pipeline): Complete model pipeline including preprocessing and model.
        
    Note:
        This is an abstract base class that should be subclassed to implement specific
        model types. Subclasses must implement at least the _create_pipeline() method.
    """
    
    def __init__(
        self,
        model_type: str,
        feature_selection_percentile: int = 50,
        random_state: int = 123,
        model_dir: Optional[str] = None
    ):
        """Initialize a Model instance with configuration parameters.
        
        This constructor sets up the initial state of the model with the specified
        configuration parameters. It doesn't create the actual model pipeline yet;
        that happens when the fit method is called.
        
        Args:
            model_type (str): Identifier string for the type of model implementation
                (e.g., "lr" for linear regression, "ridge" for Ridge regression).
            feature_selection_percentile (int, optional): Percentage of features to keep
                based on their mutual information scores with the target. Lower values
                mean more aggressive feature selection. Defaults to 50.
            random_state (int, optional): Seed for random number generation to ensure
                reproducibility across runs. Defaults to 123.
            model_dir (str, optional): Directory path where model artifacts should be
                saved and loaded from. If None, uses a default path based on project root.
                Defaults to None.
        """
        self.model_type = model_type
        self.feature_selection_percentile = feature_selection_percentile
        self.random_state = random_state
        
        if model_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.model_dir = os.path.join(base_dir, "models")
        else:
            self.model_dir = model_dir
            
        self.model = None
        self.preprocessor = None
        self.pipeline = None
    
    def _create_pipeline(self) -> Pipeline:
        """Create the model pipeline with preprocessing and estimator.
        
        This method should be implemented by subclasses to create a scikit-learn
        Pipeline that combines preprocessing steps with the specific estimator
        for the model type.
        
        Returns:
            Pipeline: A scikit-learn Pipeline with preprocessing and estimator steps.
            
        Raises:
            NotImplementedError: If the subclass doesn't implement this method.
        """
        raise NotImplementedError("Subclasses must implement _create_pipeline()")
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'Model':
        """Train the model using the provided data.
        
        This method creates the model pipeline by engineering features and setting up
        the estimator, then fits the pipeline to the training data. After training,
        the model is ready for making predictions.
        
        Args:
            X (pd.DataFrame): Feature DataFrame containing the predictor variables.
            y (pd.Series): Target Series containing the resale prices to predict.
            
        Returns:
            Model: The trained model instance (self) for method chaining.
            
        Example:
            >>> model = LinearRegressionModel()
            >>> trained_model = model.fit(X_train, y_train)
            >>> predictions = trained_model.predict(X_test)
        """        # Engineer features
        # Note: We only store the preprocessor and ignore the feature lists returned by engineer_features
        self.preprocessor, _, _ = engineer_features(
            X, 
            feature_selection_percentile=self.feature_selection_percentile
        )
        
        # Create and fit pipeline
        self.pipeline = self._create_pipeline()
        self.model = self.pipeline.fit(X, y)
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data using the trained model.
        
        This method uses the trained model to generate predictions for new feature data.
        It checks if the model has been trained before attempting to make predictions.
        
        Args:
            X (pd.DataFrame): Feature DataFrame for which to generate predictions.
                Must have the same features as the training data.
                
        Returns:
            np.ndarray: Array of predicted resale prices for the input features.
            
        Raises:
            RuntimeError: If called before the model has been trained.
            
        Example:
            >>> model = LinearRegressionModel().fit(X_train, y_train)
            >>> predictions = model.predict(X_new)
            >>> print(f"Predicted price: ${predictions[0]:,.2f}")
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet. Call fit() first.")
        
        return self.model.predict(X)
    
    def score(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Calculate the R-squared score of the model on the provided dataset.
        
        This method evaluates the model's performance using the coefficient of
        determination (R-squared), which represents the proportion of variance in the
        dependent variable that is predictable from the independent variables.
        
        An R-squared score closer to 1.0 indicates a better fit of the model to the data.
        
        Args:
            X (pd.DataFrame): Feature DataFrame containing the predictor variables.
            y (pd.Series): Target Series containing the true resale prices.
            
        Returns:
            float: The R-squared score of the model on the provided data, ranging from
                negative infinity to 1.0, where 1.0 is a perfect score.
            
        Raises:
            RuntimeError: If called before the model has been trained.
            
        Example:
            >>> model = LinearRegressionModel().fit(X_train, y_train)
            >>> r2 = model.score(X_val, y_val)
            >>> print(f"R-squared Score: {r2:.4f}")
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet. Call fit() first.")
        
        return self.model.score(X, y)
    
    def save(self, filename: Optional[str] = None) -> str:
        """Save the trained model to disk.
        
        This method serializes the trained model to a pickle file for later use.
        It ensures the model directory exists before saving and generates a default
        filename based on the model type if none is provided.
        
        Args:
            filename (str, optional): Name to use for the saved model file. If None,
                generates a name based on the model type. Defaults to None.
                
        Returns:
            str: The full path to the saved model file.
            
        Raises:
            RuntimeError: If called before the model has been trained.
            
        Example:
            >>> model = LinearRegressionModel().fit(X_train, y_train)
            >>> model_path = model.save("my_linear_model.pkl")
            >>> print(f"Model saved to: {model_path}")
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet. Call fit() first.")
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Generate filename if not provided
        if filename is None:
            filename = f"pipe_{self.model_type}_ss.pkl"
        
        # Save model to file
        model_path = os.path.join(self.model_dir, filename)
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
            
        return model_path
    
    def load(self, filename: Optional[str] = None) -> 'Model':
        """Load a trained model from disk.
        
        This method deserializes a previously saved model from a pickle file.
        It sets the loaded model as the current model for this instance.
        
        Args:
            filename (str, optional): Name of the model file to load. If None,
                generates a name based on the model type. Defaults to None.
                
        Returns:
            Model: The current model instance with the loaded model for method chaining.
            
        Raises:
            FileNotFoundError: If the model file doesn't exist.
            
        Example:
            >>> model = LinearRegressionModel().load("my_linear_model.pkl")
            >>> predictions = model.predict(X_new)
        """
        # Generate filename if not provided
        if filename is None:
            filename = f"pipe_{self.model_type}_ss.pkl"
        
        # Load model from file
        model_path = os.path.join(self.model_dir, filename)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
            self.pipeline = self.model
            
        return self
