"""Linear regression models for HDB resale price prediction.

This module implements various linear regression models for predicting HDB resale
prices. It extends the base Model class with specific linear regression implementations,
including:

1. Linear Regression - Standard OLS regression without regularization
2. Ridge Regression - Linear regression with L2 regularization
3. Lasso Regression - Linear regression with L1 regularization

Each model class encapsulates the configuration, training, and evaluation of a
specific regression algorithm, while inheriting common functionality from the
base Model class.

The models in this module are designed to provide interpretability through their
coefficients, which can help identify the most important features influencing
HDB resale prices.

Typical usage:
    >>> from src.models.linear import LinearRegressionModel
    >>> from src.data.loader import load_train_test_data
    >>> X_train, X_test, y_train, y_test = load_train_test_data()
    >>> model = LinearRegressionModel()
    >>> model.train(X_train, y_train)
    >>> predictions = model.predict(X_test)
    >>> evaluation = model.evaluate(X_test, y_test)
"""
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, RidgeCV, LassoCV
from sklearn.pipeline import Pipeline

from src.models.base import Model


class LinearRegressionModel(Model):
    """Linear regression model without regularization for HDB price prediction.
    
    This class implements a standard Ordinary Least Squares (OLS) linear regression
    model for predicting HDB resale prices. It extends the base Model class and
    configures a scikit-learn LinearRegression estimator within a pipeline that
    includes preprocessing steps.
    
    The model optimizes for the lowest mean squared error and provides coefficients
    that can be interpreted as the price impact of each feature. This model serves
    as a baseline for comparison with more complex regularized models.
    
    Attributes:
        feature_selection_percentile (int): Percentage of features to keep based
            on mutual information scores.
        random_state (int): Random seed for reproducibility.
        model_dir (str, optional): Directory for saving/loading model artifacts.
        preprocessor (ColumnTransformer): Data preprocessing pipeline.
        pipeline (Pipeline): Complete model pipeline including preprocessing and regression.
        
    Example:        >>> model = LinearRegressionModel(feature_selection_percentile=75)
        >>> model.train(X_train, y_train)
        >>> r2_score = model.evaluate(X_test, y_test)["r2_score"]
    """
    
    def __init__(
        self,
        feature_selection_percentile: int = 50,
        random_state: int = 123,
        model_dir: str = None
    ):
        """Initialize a linear regression model with specified configuration.
        
        This constructor configures a linear regression model by setting parameters
        and initializing the base Model class with the appropriate model type.
        
        Args:
            feature_selection_percentile (int, optional): Percentage of features to
                keep based on mutual information with the target variable. Lower values
                produce more aggressive feature selection. Defaults to 50.
            random_state (int, optional): Seed for random number generation to ensure
                reproducibility of results. Defaults to 123.
            model_dir (str, optional): Directory path where model artifacts (fitted
                model, preprocessor, metrics) will be saved. If None, uses the default
                directory from the base Model class. Defaults to None.
                
        Note:
            The actual model pipeline isn't created until needed, but the configuration
            is stored for later use. This lazy initialization pattern helps with
            serialization and allows changing parameters before training.
        """        
        super().__init__(
            model_type="lr",
            feature_selection_percentile=feature_selection_percentile,
            random_state=random_state,
            model_dir=model_dir
        )
        
    def _create_pipeline(self) -> Pipeline:
        """Create the scikit-learn pipeline for the linear regression model.
        
        This method is called internally by the base Model class to create
        the model's processing pipeline. It combines the preprocessor (which
        handles feature transformations) with the LinearRegression estimator
        configured to use all available CPU cores.
        
        The method is overridden from the base Model class to provide the specific
        estimator configuration for standard linear regression.
        
        Returns:
            Pipeline: A scikit-learn Pipeline with preprocessing and linear regression
                steps configured according to the model's parameters.
                
        Note:
            This is a protected method called by the base class and not typically
            invoked directly by users of the class.
        """
        return Pipeline(
            steps=[
                ("preprocessor", self.preprocessor), 
                ("regressor", LinearRegression(n_jobs=-1))
            ]
        )


class RidgeRegressionModel(Model):
    """Ridge regression model with L2 regularization for HDB price prediction.
    
    This class implements a Ridge regression model for predicting HDB resale prices.
    It extends the base Model class and configures a scikit-learn RidgeCV estimator
    within a pipeline that includes preprocessing steps.
    
    Ridge regression adds an L2 penalty (sum of squared coefficients) to the loss
    function, which helps prevent overfitting and handles multicollinearity in
    the feature set. The optimal regularization strength (alpha) is selected
    automatically using cross-validation.
    
    Attributes:
        feature_selection_percentile (int): Percentage of features to keep based 
            on mutual information scores.
        alphas (np.ndarray): Array of alpha values to try during cross-validation.
        cv (int): Number of cross-validation folds.
        random_state (int): Random seed for reproducibility.
        model_dir (str, optional): Directory for saving/loading model artifacts.
        preprocessor (ColumnTransformer): Data preprocessing pipeline.
        pipeline (Pipeline): Complete model pipeline including preprocessing and regression.
        
    Example:
        >>> model = RidgeRegressionModel(feature_selection_percentile=75)
        >>> model.train(X_train, y_train)
        >>> print(f"Best alpha: {model.best_alpha}")
        >>> r2_score = model.evaluate(X_test, y_test)["r2_score"]
    """
    
    def __init__(
        self,
        feature_selection_percentile: int = 50,
        alphas: np.ndarray = None,
        cv: int = 5,
        random_state: int = 123,
        model_dir: str = None
    ):
        """Initialize a ridge regression model with specified configuration.
        
        This constructor configures a Ridge regression model with cross-validation
        for automatic selection of the optimal regularization strength (alpha).
        It sets up the model parameters and initializes the base Model class.
        
        Args:
            feature_selection_percentile (int, optional): Percentage of features to
                keep based on mutual information with the target variable. Lower values
                produce more aggressive feature selection. Defaults to 50.
            alphas (np.ndarray, optional): Array of alpha values to try during
                cross-validation. If None, uses a default range of values appropriate
                for most HDB price prediction tasks. Defaults to None.
            cv (int, optional): Number of cross-validation folds to use when selecting
                the optimal alpha value. Higher values give more accurate estimates but
                increase computation time. Defaults to 5.
            random_state (int, optional): Seed for random number generation to ensure
                reproducibility of results. Defaults to 123.
            model_dir (str, optional): Directory path where model artifacts (fitted
                model, preprocessor, metrics) will be saved. If None, uses the default
                directory from the base Model class. Defaults to None.
                
        Note:
            Ridge regression adds L2 regularization to linear regression, which helps
            prevent overfitting when there are many features or multicollinearity.
        """
        if alphas is None:
            alphas = np.logspace(-3, 3, 10)
            
        super().__init__(
            model_type="ridge",
            feature_selection_percentile=feature_selection_percentile,
            random_state=random_state,
            model_dir=model_dir
        )
        
        self.alphas = alphas
        self.cv = cv
    
    def _create_pipeline(self) -> Pipeline:
        """Create the scikit-learn pipeline for the ridge regression model.
        
        This method is called internally by the base Model class to create
        the model's processing pipeline. It combines the preprocessor (which
        handles feature transformations) with the RidgeCV estimator that
        automatically selects the optimal regularization strength (alpha)
        from the provided range using cross-validation.
        
        The method is overridden from the base Model class to provide the specific
        estimator configuration for ridge regression with cross-validation.
        
        Returns:
            Pipeline: A scikit-learn Pipeline with preprocessing and ridge regression
                steps configured according to the model's parameters.
                
        Note:
            This is a protected method called by the base class and not typically
            invoked directly by users of the class.
        """
        return Pipeline(
            steps=[
                ("preprocessor", self.preprocessor), 
                ("regressor", RidgeCV(
                    alphas=self.alphas,
                    cv=self.cv,
                    scoring='neg_mean_squared_error'
                ))
            ]
        )
        
    @property
    def best_alpha(self) -> float:
        """Get the optimal alpha value selected through cross-validation.
        
        This property provides access to the optimal regularization strength (alpha)
        that was selected by RidgeCV during training through cross-validation.
        The alpha value represents the strength of the L2 penalty; higher values
        create simpler models with smaller coefficients.
        
        Returns:
            float: The optimal alpha value selected during model training.
            
        Raises:
            RuntimeError: If accessed before the model has been trained.
            
        Example:
            >>> model = RidgeRegressionModel()
            >>> model.fit(X_train, y_train)
            >>> print(f"Best alpha selected: {model.best_alpha:.6f}")
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet. Call fit() first.")
            
        # Get the regressor from the pipeline
        ridge_cv = self.model.named_steps['regressor']
        return ridge_cv.alpha_


class LassoRegressionModel(Model):
    """Lasso regression model with L1 regularization for HDB price prediction.
    
    This class implements a Lasso regression model for predicting HDB resale prices.
    It extends the base Model class and configures a scikit-learn LassoCV estimator
    within a pipeline that includes preprocessing steps.
    
    Lasso regression adds an L1 penalty (sum of absolute coefficients) to the loss
    function, which helps with feature selection by potentially setting some coefficients
    to exactly zero. The optimal regularization strength (alpha) is selected
    automatically using cross-validation.
    
    Attributes:
        feature_selection_percentile (int): Percentage of features to keep based
            on mutual information scores.
        n_alphas (int): Number of alpha values to try in cross-validation.
        cv (int): Number of cross-validation folds.
        random_state (int): Random seed for reproducibility.
        model_dir (str, optional): Directory for saving/loading model artifacts.
        preprocessor (ColumnTransformer): Data preprocessing pipeline.
        pipeline (Pipeline): Complete model pipeline including preprocessing and regression.
        
    Example:
        >>> model = LassoRegressionModel(feature_selection_percentile=75)
        >>> model.fit(X_train, y_train)
        >>> print(f"Best alpha: {model.best_alpha}")
        >>> r2_score = model.evaluate(X_test, y_test)["r2_score"]
        >>> print(f"Non-zero features: {model.n_nonzero_features}")
    """
    
    def __init__(
        self,
        feature_selection_percentile: int = 50,
        n_alphas: int = 100,
        cv: int = 5,
        random_state: int = 123,
        model_dir: str = None
    ):
        """Initialize a lasso regression model with specified configuration.
        
        This constructor configures a Lasso regression model with cross-validation
        for automatic selection of the optimal regularization strength (alpha).
        It sets up the model parameters and initializes the base Model class.
        
        Args:
            feature_selection_percentile (int, optional): Percentage of features to
                keep based on mutual information with the target variable. Lower values
                produce more aggressive feature selection. Defaults to 50.
            n_alphas (int, optional): Number of alpha values to try during cross-validation.
                Higher values provide finer-grained optimization but increase computation time.
                Defaults to 100.
            cv (int, optional): Number of cross-validation folds to use when selecting
                the optimal alpha value. Higher values give more accurate estimates but
                increase computation time. Defaults to 5.
            random_state (int, optional): Seed for random number generation to ensure
                reproducibility of results. Defaults to 123.
            model_dir (str, optional): Directory path where model artifacts (fitted
                model, preprocessor, metrics) will be saved. If None, uses the default
                directory from the base Model class. Defaults to None.
                
        Note:
            Lasso regression adds L1 regularization to linear regression, which helps
            with feature selection by potentially setting some coefficients to exactly zero.
        """
        super().__init__(
            model_type="lasso",
            feature_selection_percentile=feature_selection_percentile,
            random_state=random_state,
            model_dir=model_dir
        )
        
        self.n_alphas = n_alphas
        self.cv = cv
    
    def _create_pipeline(self) -> Pipeline:
        """Create the scikit-learn pipeline for the lasso regression model.
        
        This method is called internally by the base Model class to create
        the model's processing pipeline. It combines the preprocessor (which
        handles feature transformations) with the LassoCV estimator that
        automatically selects the optimal regularization strength (alpha)
        using cross-validation.
        
        The method is overridden from the base Model class to provide the specific
        estimator configuration for lasso regression with cross-validation.
        
        Returns:
            Pipeline: A scikit-learn Pipeline with preprocessing and lasso regression
                steps configured according to the model's parameters.
                
        Note:
            This is a protected method called by the base class and not typically
            invoked directly by users of the class.
        """
        return Pipeline(
            steps=[
                ("preprocessor", self.preprocessor), 
                ("regressor", LassoCV(
                    n_alphas=self.n_alphas,
                    cv=self.cv,
                    random_state=self.random_state,
                    n_jobs=-1
                ))
            ]
        )
        
    @property
    def best_alpha(self) -> float:
        """Get the optimal alpha value selected through cross-validation.
        
        This property provides access to the optimal regularization strength (alpha)
        that was selected by LassoCV during training through cross-validation.
        The alpha value represents the strength of the L1 penalty; higher values
        create simpler models with more coefficients set to zero.
        
        Returns:
            float: The optimal alpha value selected during model training.
            
        Raises:
            RuntimeError: If accessed before the model has been trained.
            
        Example:
            >>> model = LassoRegressionModel()
            >>> model.fit(X_train, y_train)
            >>> print(f"Best alpha selected: {model.best_alpha:.6f}")
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet. Call fit() first.")
            
        # Get the regressor from the pipeline
        lasso_cv = self.model.named_steps['regressor']
        return lasso_cv.alpha_
        
    @property
    def n_nonzero_features(self) -> int:
        """Get the number of features with non-zero coefficients.
        
        This property returns the count of features that have non-zero coefficients
        in the trained Lasso model. It provides insight into how aggressive the
        feature selection has been - fewer non-zero features indicates stronger
        regularization and a simpler model.
        
        Returns:
            int: The number of features with non-zero coefficients.
            
        Raises:
            RuntimeError: If accessed before the model has been trained.
            
        Example:
            >>> model = LassoRegressionModel()
            >>> model.fit(X_train, y_train)
            >>> print(f"Number of features used: {model.n_nonzero_features}")
            >>> print(f"Out of total features: {X_train.shape[1]}")
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet. Call fit() first.")
            
        # Get the regressor from the pipeline
        lasso_cv = self.model.named_steps['regressor']
        return np.sum(lasso_cv.coef_ != 0)
