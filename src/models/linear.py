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
from src.utils.helpers import load_config


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
        
    Example:
        >>> model = LinearRegressionModel(feature_selection_percentile=75)
        >>> model.train(X_train, y_train)
        >>> r2_score = model.evaluate(X_test, y_test)["r2_score"]
    """
    
    def __init__(
        self,
        feature_selection_percentile: int = None,
        random_state: int = None,
        model_dir: str = None,
        fit_intercept: bool = None,
        normalize: bool = None,
        config: dict = None
    ):
        """Initialize a linear regression model with specified configuration.
        
        This constructor configures a linear regression model by setting parameters
        and initializing the base Model class with the appropriate model type.
        
        Args:
            feature_selection_percentile (int, optional): Percentage of features to
                keep based on mutual information with the target variable. Lower values
                produce more aggressive feature selection. Defaults to None, uses config value.
            random_state (int, optional): Seed for random number generation to ensure
                reproducibility of results. Defaults to None, uses config value.
            model_dir (str, optional): Directory path where model artifacts (fitted
                model, preprocessor, metrics) will be saved. If None, uses the default
                directory from the base Model class. Defaults to None.
            fit_intercept (bool, optional): Whether to calculate the intercept for this model.
                Defaults to None, uses config value.
            normalize (bool, optional): Whether to normalize features. 
                Defaults to None, uses config value.
            config (dict, optional): Model configuration dictionary. If provided, overrides
                other parameters with values from configuration. Defaults to None.
                
        Note:
            The actual model pipeline isn't created until needed, but the configuration
            is stored for later use. This lazy initialization pattern helps with
            serialization and allows changing parameters before training.
        """
        # Load configuration if not provided
        if config is None:
            config = load_config('model_config')
        
        # Get evaluation settings
        eval_config = config.get('evaluation', {})
        if random_state is None:
            random_state = eval_config.get('random_state', 42)
            
        # Get feature selection settings
        feature_config = config.get('features', {}).get('feature_selection', {})
        if feature_selection_percentile is None:
            feature_selection_percentile = 50  # Default
          # Get model specific settings
        model_config = config.get('models', {}).get('linear_regression', {})
        self.fit_intercept = fit_intercept if fit_intercept is not None else model_config.get('fit_intercept', True)
        # The 'normalize' parameter is deprecated in scikit-learn and will be removed
        # It's kept here for config compatibility but won't be used
        self._normalize_deprecated = normalize if normalize is not None else model_config.get('normalize', False)
        
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
        configured using the parameters from the configuration file.
        
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
                ("regressor", LinearRegression(
                    fit_intercept=self.fit_intercept,
                    n_jobs=-1
                ))
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
        feature_selection_percentile: int = None,
        alpha: float = None,
        fit_intercept: bool = None,
        normalize: bool = None,
        max_iter: int = None,
        tol: float = None,
        solver: str = None,
        random_state: int = None,
        model_dir: str = None,
        config: dict = None
    ):
        """Initialize a ridge regression model with specified configuration.
        
        This constructor configures a Ridge regression model with configurable
        regularization strength (alpha). It sets up the model parameters and 
        initializes the base Model class.
        
        Args:
            feature_selection_percentile (int, optional): Percentage of features to
                keep based on mutual information with the target variable. Lower values
                produce more aggressive feature selection. Defaults to None, uses config value.
            alpha (float, optional): Regularization strength parameter. Larger values
                specify stronger regularization. Defaults to None, uses config value.
            fit_intercept (bool, optional): Whether to calculate the intercept for this model.
                Defaults to None, uses config value.
            normalize (bool, optional): Whether to normalize features.
                Defaults to None, uses config value.
            max_iter (int, optional): Maximum number of iterations for solver.
                Defaults to None, uses config value.
            tol (float, optional): Tolerance for stopping criteria.
                Defaults to None, uses config value.
            solver (str, optional): Solver to use for the optimization.
                Defaults to None, uses config value.
            random_state (int, optional): Seed for random number generation to ensure
                reproducibility of results. Defaults to None, uses config value.
            model_dir (str, optional): Directory path where model artifacts (fitted
                model, preprocessor, metrics) will be saved. If None, uses the default
                directory from the base Model class. Defaults to None.
            config (dict, optional): Model configuration dictionary. If provided, overrides
                other parameters with values from configuration. Defaults to None.
                
        Note:
            Ridge regression adds L2 regularization to linear regression, which helps
            prevent overfitting when there are many features or multicollinearity.
        """
        # Load configuration if not provided
        if config is None:
            config = load_config('model_config')
        
        # Get evaluation settings
        eval_config = config.get('evaluation', {})
        if random_state is None:
            random_state = eval_config.get('random_state', 42)
            
        # Get feature selection settings
        feature_config = config.get('features', {}).get('feature_selection', {})
        if feature_selection_percentile is None:
            feature_selection_percentile = 50  # Default
        
        # Get model specific settings
        model_config = config.get('models', {}).get('ridge_regression', {})
        self.alpha = alpha if alpha is not None else model_config.get('alpha', 1.0)
        self.fit_intercept = fit_intercept if fit_intercept is not None else model_config.get('fit_intercept', True)
        # The 'normalize' parameter is deprecated in scikit-learn and will be removed
        self._normalize_deprecated = normalize if normalize is not None else model_config.get('normalize', False)
        self.max_iter = max_iter if max_iter is not None else model_config.get('max_iter', 1000)
        self.tol = tol if tol is not None else model_config.get('tol', 0.001)
        self.solver = solver if solver is not None else model_config.get('solver', 'auto')
            
        super().__init__(
            model_type="ridge",
            feature_selection_percentile=feature_selection_percentile,
            random_state=random_state,
            model_dir=model_dir
        )
    def _create_pipeline(self) -> Pipeline:
        """Create the scikit-learn pipeline for the ridge regression model.
        
        This method is called internally by the base Model class to create
        the model's processing pipeline. It combines the preprocessor (which
        handles feature transformations) with the Ridge estimator that uses
        the configured hyperparameters from the model_config.yaml file.
        
        The method is overridden from the base Model class to provide the specific
        estimator configuration for ridge regression.
        
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
                ("regressor", Ridge(
                    alpha=self.alpha,
                    fit_intercept=self.fit_intercept,
                    # normalize parameter removed as it's deprecated in scikit-learn
                    max_iter=self.max_iter,
                    tol=self.tol,
                    solver=self.solver,
                    random_state=self.random_state
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
    It extends the base Model class and configures a scikit-learn Lasso estimator
    within a pipeline that includes preprocessing steps.
    
    Lasso regression adds an L1 penalty (sum of absolute coefficients) to the loss
    function, which helps with feature selection by potentially setting some coefficients
    to exactly zero.
    
    Attributes:
        feature_selection_percentile (int): Percentage of features to keep based
            on mutual information scores.
        alpha (float): L1 regularization strength parameter.
        fit_intercept (bool): Whether to fit the intercept.
        normalize (bool): Whether to normalize the data.
        max_iter (int): Maximum number of iterations.
        tol (float): Tolerance for stopping criteria.
        warm_start (bool): Whether to reuse previous solution.
        selection (str): Feature selection method ('cyclic' or 'random').
        random_state (int): Random seed for reproducibility.
        model_dir (str, optional): Directory for saving/loading model artifacts.
        
    Example:
        >>> model = LassoRegressionModel()
        >>> model.fit(X_train, y_train)
        >>> r2_score = model.evaluate(X_test, y_test)["r2_score"]
        >>> print(f"Non-zero features: {model.n_nonzero_features}")
    """
    
    def __init__(
        self,
        feature_selection_percentile: int = None,
        alpha: float = None,
        fit_intercept: bool = None,
        normalize: bool = None,
        max_iter: int = None,
        tol: float = None,
        warm_start: bool = None,
        selection: str = None,
        random_state: int = None,
        model_dir: str = None,
        config: dict = None
    ):
        """Initialize a lasso regression model with specified configuration.
        
        This constructor configures a Lasso regression model with parameters loaded
        from the configuration file. It sets up the model parameters and initializes 
        the base Model class.
        
        Args:
            feature_selection_percentile (int, optional): Percentage of features to
                keep based on mutual information with the target variable. Lower values
                produce more aggressive feature selection. Defaults to None, uses config value.
            alpha (float, optional): Regularization strength parameter. Larger values specify
                stronger regularization. Defaults to None, uses config value.
            fit_intercept (bool, optional): Whether to calculate the intercept for this model.
                Defaults to None, uses config value.
            normalize (bool, optional): Whether to normalize features.
                Defaults to None, uses config value.
            max_iter (int, optional): Maximum number of iterations for solver. 
                Defaults to None, uses config value.
            tol (float, optional): Tolerance for stopping criteria. Defaults to None, uses config value.
            warm_start (bool, optional): Whether to reuse previous solution. 
                Defaults to None, uses config value.
            selection (str, optional): Selection strategy among 'cyclic', 'random'. 
                Defaults to None, uses config value.
            random_state (int, optional): Seed for random number generation to ensure
                reproducibility of results. Defaults to None, uses config value.
            model_dir (str, optional): Directory path where model artifacts (fitted
                model, preprocessor, metrics) will be saved. If None, uses the default
                directory from the base Model class. Defaults to None.
            config (dict, optional): Model configuration dictionary. If provided, overrides
                other parameters with values from configuration. Defaults to None.
                
        Note:
            Lasso regression adds L1 regularization to linear regression, which helps
            with feature selection by potentially setting some coefficients to exactly zero.
        """
        # Load configuration if not provided
        if config is None:
            config = load_config('model_config')
        
        # Get evaluation settings
        eval_config = config.get('evaluation', {})
        if random_state is None:
            random_state = eval_config.get('random_state', 42)
            
        # Get feature selection settings
        feature_config = config.get('features', {}).get('feature_selection', {})
        if feature_selection_percentile is None:
            feature_selection_percentile = 50  # Default
        
        # Get model specific settings
        model_config = config.get('models', {}).get('lasso_regression', {})
        self.alpha = alpha if alpha is not None else model_config.get('alpha', 0.01)
        self.fit_intercept = fit_intercept if fit_intercept is not None else model_config.get('fit_intercept', True)
        # The 'normalize' parameter is deprecated in scikit-learn and will be removed
        self._normalize_deprecated = normalize if normalize is not None else model_config.get('normalize', False)
        self.max_iter = max_iter if max_iter is not None else model_config.get('max_iter', 1000)
        self.tol = tol if tol is not None else model_config.get('tol', 0.0001)
        self.warm_start = warm_start if warm_start is not None else model_config.get('warm_start', False)
        self.selection = selection if selection is not None else model_config.get('selection', 'cyclic')
        
        super().__init__(
            model_type="lasso",
            feature_selection_percentile=feature_selection_percentile,
            random_state=random_state,
            model_dir=model_dir
        )
    
    def _create_pipeline(self) -> Pipeline:
        """Create the scikit-learn pipeline for the lasso regression model.
        
        This method is called internally by the base Model class to create
        the model's processing pipeline. It combines the preprocessor (which
        handles feature transformations) with the Lasso estimator that uses
        the configured hyperparameters from the model_config.yaml file.
        
        The method is overridden from the base Model class to provide the specific
        estimator configuration for lasso regression.
        
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
                ("regressor", Lasso(
                    alpha=self.alpha,
                    fit_intercept=self.fit_intercept,
                    # normalize parameter removed as it's deprecated in scikit-learn
                    max_iter=self.max_iter,
                    tol=self.tol,
                    warm_start=self.warm_start,
                    selection=self.selection,
                    random_state=self.random_state
                ))
            ]
        )
    
    @property
    def alpha_value(self) -> float:
        """Get the alpha value used in the model.
        
        This property provides access to the regularization strength (alpha)
        that was used for the Lasso model. The alpha value represents the 
        strength of the L1 penalty; higher values create simpler models 
        with more coefficients set to zero.
        
        Returns:
            float: The alpha value configured for this model.
            
        Example:
            >>> model = LassoRegressionModel()
            >>> print(f"Alpha value: {model.alpha_value:.6f}")
        """
        return self.alpha
        
    @property
    def n_nonzero_features(self) -> int:
        """Get the number of features with non-zero coefficients.
        
        This property returns the count of features that have non-zero coefficients
        in the trained Lasso model. It provides insight into how aggressive the
        feature selection has been - fewer non-zero features indicates stronger
        regularization that has eliminated more features.
        
        Returns:
            int: Number of feature coefficients that are non-zero.
            
        Raises:
            RuntimeError: If accessed before the model has been trained.
            
        Example:
            >>> model = LassoRegressionModel()
            >>> model.fit(X_train, y_train)
            >>> print(f"Non-zero features: {model.n_nonzero_features}")
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained yet. Call fit() first.")
            
        # Get the regressor from the pipeline
        lasso = self.model.named_steps['regressor']
        
        # Count non-zero coefficients
        return np.sum(lasso.coef_ != 0)
