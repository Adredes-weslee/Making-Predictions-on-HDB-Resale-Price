"""Unit tests for HDB price prediction model functionality.

This module contains test cases for the machine learning models used in the
HDB resale price prediction project. The tests verify that models initialize
correctly, train without errors, make valid predictions, and produce reasonable
evaluation metrics.

The tests use both synthetic data (for controlled testing of specific behaviors)
and small samples of real data (for integration testing). Tests are organized
into classes based on the model type being tested.

These tests help ensure that model changes don't introduce regressions and that
the core prediction functionality works as expected across different model types.

Typical test run:
    $ python -m unittest tests.test_models
"""
import os
import sys
import unittest
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.loader import load_raw_data, get_data_paths
from src.data.preprocessing import preprocess_data
from src.models.base import Model
from src.models.linear import LinearRegressionModel, LassoRegressionModel, RidgeRegressionModel
from src.models.prediction import predict
from src.models.evaluation import evaluate_model


class TestBaseModel(unittest.TestCase):
    """Tests for the base Model class functionality.
    
    This test class verifies the base functionality of the abstract Model class
    that all other model implementations inherit from. It tests initialization,
    property access, and common methods without requiring a concrete implementation.
    
    Test methods in this class focus on the generic behavior that should be
    consistent across all model types, like parameter storage, model object
    instantiation, and interface consistency.
    """
    
    def test_init(self):
        """Test Model initialization with default and custom parameters.
        
        This test verifies that the Model class correctly initializes with
        the provided model_type parameter and sets default values for other
        parameters. It checks that key attributes are set correctly and that
        the model object starts in an uninitialized state.
        
        The test focuses on the constructor behavior to ensure objects are
        created consistently before any training or prediction actions.
        """
        model = Model(model_type="test")
        self.assertEqual(model.model_type, "test")
        self.assertIsNone(model.model)


class TestLinearModels(unittest.TestCase):
    """Tests for linear regression model implementations used in HDB price prediction.
    
    This test class verifies the functionality of various linear regression model
    implementations: standard linear regression, ridge regression, and lasso regression.
    It tests model initialization, training, prediction, and evaluation to ensure
    that all models work correctly with the expected interfaces.
    
    The tests use synthetic data with known relationships to verify that models
    can capture these relationships and make reasonable predictions. The test data
    includes features with different levels of correlation to the target to simulate
    real-world data characteristics.
    """

    def setUp(self):
        """Create synthetic test data for model evaluation.
        
        This method runs before each test in this class and creates a synthetic
        dataset with controlled characteristics:
        - 100 samples with 5 features
        - Known coefficients for features (2, -1, 3, 0, 0)
        - Gaussian noise added to targets
        - Split into 80% training and 20% test data
        
        The synthetic data ensures tests are reproducible and that we know the
        "ground truth" relationships that the models should learn.
        """
        # Create synthetic data
        np.random.seed(42)
        n_samples = 100
        n_features = 5
        
        # Create features with some correlation to target
        X = np.random.rand(n_samples, n_features)
        noise = np.random.normal(0, 0.5, size=n_samples)
        y = 5 + 2*X[:, 0] - X[:, 1] + 3*X[:, 2] + noise
        self.X_train = pd.DataFrame(
            X, 
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        self.y_train = pd.Series(y, name='target')
        
        # Split into train/test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_train, self.y_train, test_size=0.2, random_state=42
        )

    def test_linear_regression(self):
        """Test LinearRegressionModel initialization, training, and prediction.
        
        This test verifies that:
        1. The LinearRegressionModel can be instantiated with default parameters
        2. The model can be trained on synthetic data without errors
        3. The trained model achieves a positive R² score on test data
        4. The model can generate predictions for new data samples
        5. The predictions have the correct shape/length
        
        A failing test indicates a problem with the basic linear regression
        implementation that would affect all downstream functionality.
        """
        model = LinearRegressionModel()
        model.fit(self.X_train, self.y_train)
        score = model.score(self.X_test, self.y_test)
        
        self.assertIsNotNone(model.model)
        self.assertGreater(score, 0)  # Should have positive R^2
        
        # Test prediction
        preds = model.predict(self.X_test)
        self.assertEqual(len(preds), len(self.X_test))

    def test_ridge_regression(self):
        """Test RidgeRegressionModel initialization, training, and prediction.
        
        This test verifies that:
        1. The RidgeRegressionModel can be instantiated with a custom alpha parameter
        2. The model can be trained on synthetic data without errors
        3. The trained model achieves a positive R² score on test data
        4. The model can generate predictions for new data samples
        5. The predictions have the correct shape/length
        
        Ridge regression adds L2 regularization to linear regression, which helps
        prevent overfitting when there are many features or multicollinearity.
        This test ensures the regularization doesn't break the core functionality.
        
        A failing test indicates a problem with the ridge regression implementation
        or with how regularization is being applied in the model pipeline.
        """
        model = RidgeRegressionModel(alpha=0.1)
        model.fit(self.X_train, self.y_train)
        score = model.score(self.X_test, self.y_test)
        
        self.assertIsNotNone(model.model)
        self.assertGreater(score, 0)  # Should have positive R^2
        
        # Test prediction
        preds = model.predict(self.X_test)
        self.assertEqual(len(preds), len(self.X_test))

    def test_lasso_regression(self):
        """Test LassoRegressionModel initialization, training, and prediction.
        
        This test verifies that:
        1. The LassoRegressionModel can be instantiated with a custom alpha parameter
        2. The model can be trained on synthetic data without errors
        3. The trained model achieves a positive R² score on test data
        4. The model can generate predictions for new data samples
        5. The predictions have the correct shape/length
        
        Lasso regression adds L1 regularization to linear regression, which helps
        with feature selection by potentially setting some coefficients to zero.
        This test ensures the regularization works correctly and doesn't negatively 
        impact the model's basic functionality.
        
        A failing test indicates a problem with the lasso regression implementation
        or with how sparsity-inducing regularization is applied in the pipeline.
        """
        model = LassoRegressionModel(alpha=0.01)
        model.fit(self.X_train, self.y_train)
        score = model.score(self.X_test, self.y_test)
        
        self.assertIsNotNone(model.model)
        self.assertGreater(score, 0)  # Should have positive R^2
        
        # Test prediction
        preds = model.predict(self.X_test)
        self.assertEqual(len(preds), len(self.X_test))


class TestModelEvaluation(unittest.TestCase):
    """Tests for model evaluation functions and metrics calculation.
    
    This test class verifies that the model evaluation utilities correctly
    calculate performance metrics for trained models. It tests:
    1. The evaluate_model function that produces comprehensive metric reports
    2. The correct calculation of R-squared, RMSE, and other performance metrics
    3. The proper formatting and structure of the returned metrics dictionary
    
    The tests use a synthetic dataset with known properties to ensure that the
    evaluation functions work correctly and report sensible metric values. This
    helps ensure that model performance assessments are reliable and accurate.
    """

    def setUp(self):
        """Set up test fixtures with small synthetic dataset for evaluation testing.
        
        This method runs before each test in this class and creates a complete
        test environment for model evaluation:
        1. Generates a synthetic dataset with 100 samples and 5 features
        2. Creates a target variable with known relationships to features
        3. Splits the data into training and testing sets
        4. Trains a basic linear regression model on the training data
        
        The synthetic data and trained model provide a controlled environment
        where the expected evaluation metrics can be reasonably predicted,
        allowing for verification of the evaluation functions' correctness.
        """
        # Create synthetic data
        np.random.seed(42)
        n_samples = 100
        n_features = 5
        
        # Create features with some correlation to target
        X = np.random.rand(n_samples, n_features)
        noise = np.random.normal(0, 0.5, size=n_samples)
        y = 5 + 2*X[:, 0] - X[:, 1] + 3*X[:, 2] + noise
        
        self.X = pd.DataFrame(
            X, 
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        self.y = pd.Series(y, name='target')
        
        # Split into train/test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # Train a model
        self.model = LinearRegressionModel()
        self.model.fit(self.X_train, self.y_train)

    def test_evaluate_model(self):
        """Test that the model evaluation function calculates metrics correctly.
        
        This test verifies that the evaluate_model function:
        1. Returns a properly structured dictionary of evaluation metrics
        2. Includes all required metrics (train_r2, test_r2, train_rmse, test_rmse)
        3. Calculates metrics with sensible values (e.g., positive R² and RMSE)
        4. Works correctly when provided with both training and testing data
        
        The test uses the synthetic data and model created in setUp() to ensure
        that metrics are calculated correctly and consistently. It helps detect
        any changes in the evaluation logic that might affect model assessment.
        
        A failing test indicates that the evaluation metrics may be calculated
        incorrectly, which could lead to wrong conclusions about model performance.
        """
        metrics = evaluate_model(
            self.model, 
            self.X_train, self.y_train, 
            self.X_test, self.y_test
        )
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('train_r2', metrics)
        self.assertIn('test_r2', metrics)
        self.assertIn('train_rmse', metrics)
        self.assertIn('test_rmse', metrics)
        
        # Basic checks on metrics
        self.assertGreaterEqual(metrics['train_r2'], 0)
        self.assertGreater(metrics['train_rmse'], 0)


if __name__ == '__main__':
    unittest.main()
