"""Module for evaluating and comparing HDB price prediction models.

This module provides functions for assessing the performance of machine learning
models for HDB resale price prediction. It includes functions for calculating
common evaluation metrics, generating performance visualizations, and comparing
multiple models side-by-side.

The evaluation functions support both single-model assessment and comparative
analysis across multiple models to help determine which approaches perform best
for the HDB resale price prediction task.

Key evaluation metrics include:
- R² (coefficient of determination)
- RMSE (root mean squared error)
- MAE (mean absolute error)
- Residual analysis metrics and visualizations

Typical usage:
    >>> from src.models.evaluation import evaluate_model
    >>> from src.models.linear import LinearRegressionModel
    >>> from src.data.loader import load_train_test_data
    >>> X_train, X_test, y_train, y_test = load_train_test_data()
    >>> model = LinearRegressionModel()
    >>> model.train(X_train, y_train)
    >>> metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
    >>> print(f"Test R²: {metrics['test_r2']:.4f}")
"""
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score

from src.models.base import Model


def evaluate_model(
    model: Model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, float]:
    """Evaluate a trained model on both training and test datasets.
    
    This function calculates key performance metrics to assess how well the
    model fits the training data and generalizes to unseen test data. It
    evaluates both goodness-of-fit (R²) and error magnitude (RMSE) for a
    comprehensive performance assessment.
    
    The comparison between training and test metrics helps identify potential
    overfitting or underfitting issues:
    - Similar performance on train and test sets suggests good generalization
    - Much better training performance than test suggests overfitting
    - Poor performance on both sets suggests underfitting or missing features
    
    Args:
        model (Model): A trained model implementing the base Model interface
            with score() and predict() methods.
        X_train (pd.DataFrame): Training feature data used to fit the model.
        y_train (pd.Series): Training target values (actual resale prices).
        X_test (pd.DataFrame): Test feature data not used during training.
        y_test (pd.Series): Test target values for evaluating predictions.
        
    Returns:
        Dict[str, float]: Dictionary containing the following metrics:
            - train_r2: R² score on training data
            - test_r2: R² score on test data
            - train_rmse: Root mean squared error on training data (in SGD)
            - test_rmse: Root mean squared error on test data (in SGD)
            
    Example:
        >>> metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
        >>> print(f"Test RMSE: ${metrics['test_rmse']:,.2f}")
        Test RMSE: $32,150.25
    """
    # Calculate scores
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    # Calculate RMSE
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)
    
    train_rmse = np.sqrt(np.mean((y_train - train_preds) ** 2))
    test_rmse = np.sqrt(np.mean((y_test - test_preds) ** 2))
    
    return {
        "train_r2": train_score,
        "test_r2": test_score,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse
    }


def cross_validate_model(
    model: Model,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int = 5,
    scoring: str = "r2"
) -> Dict[str, float]:
    """Perform cross-validation on a model to estimate generalization performance.
    
    This function implements k-fold cross-validation to provide a more robust
    estimate of model performance than a single train/test split. The data is
    divided into 'cv' folds, and the model is trained and evaluated 'cv' times,
    each time using a different fold as the validation set.
    
    Cross-validation helps detect overfitting and provides a more reliable
    estimate of how the model will perform on unseen data. The function calculates
    both the mean and standard deviation of the scoring metric across all folds.
    
    Args:
        model (Model): An untrained model implementing the base Model interface.
            The model will be fit multiple times during cross-validation.
        X (pd.DataFrame): Feature data to use for cross-validation.
        y (pd.Series): Target values (actual resale prices) for cross-validation.
        cv (int, optional): Number of cross-validation folds. Higher values give
            more reliable estimates but increase computation time. Defaults to 5.
        scoring (str, optional): Scoring metric to use for evaluation. 
            Options include "r2", "neg_mean_squared_error", "neg_mean_absolute_error".
            Defaults to "r2".
            
    Returns:
        Dict[str, float]: Dictionary containing cross-validation results:
            - mean_score: Mean value of the scoring metric across all folds
            - std_score: Standard deviation of the scoring metric across all folds
            
    Example:
        >>> cv_results = cross_validate_model(
        ...     LinearRegressionModel(), X, y, cv=10, scoring="r2"
        ... )
        >>> print(f"Mean R²: {cv_results['mean_score']:.4f} ± {cv_results['std_score']:.4f}")
        Mean R²: 0.8534 ± 0.0213
    """
    # Create a copy of the model to avoid modifying the original
    model_copy = type(model)()
    
    # Perform cross-validation
    scores = cross_val_score(
        model_copy, X, y, 
        cv=cv, 
        scoring=scoring,
        n_jobs=-1  # Use all available cores for parallel processing
    )
    
    # Return the mean and standard deviation of scores
    return {
        "mean_score": np.mean(scores),
        "std_score": np.std(scores)
    }


def compare_models(
    models: Dict[str, Model],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> pd.DataFrame:
    """Compare multiple models on the same dataset with standardized metrics.
    
    This function evaluates multiple models on the same training and test data, 
    collecting performance metrics for each model to facilitate direct comparison.
    It's useful for model selection and for understanding the tradeoffs between
    different model types or configurations.
    
    The comparison includes both training and test performance to help identify
    overfitting or underfitting issues across the model set.
    
    Args:
        models (Dict[str, Model]): Dictionary mapping model names to trained model
            objects implementing the Model interface.
        X_train (pd.DataFrame): Training feature data used when fitting the models.
        y_train (pd.Series): Training target values (actual resale prices).
        X_test (pd.DataFrame): Test feature data not seen during training.
        y_test (pd.Series): Test target values for evaluating predictions.
        
    Returns:
        pd.DataFrame: DataFrame with rows for each model and columns for different
            performance metrics. The metrics include:
            - model_name: Name of the model (from the dictionary keys)
            - train_r2: R² score on training data
            - test_r2: R² score on test data
            - train_rmse: Root mean squared error on training data
            - test_rmse: Root mean squared error on test data
            - test_train_ratio: Ratio of test RMSE to train RMSE (overfitting indicator)
            
    Example:
        >>> models = {
        ...     'linear': LinearRegressionModel().fit(X_train, y_train),
        ...     'ridge': RidgeRegressionModel().fit(X_train, y_train),
        ...     'lasso': LassoRegressionModel().fit(X_train, y_train)
        ... }
        >>> comparison = compare_models(models, X_train, y_train, X_test, y_test)
        >>> print(comparison.sort_values('test_rmse'))
    """
    results = []
    
    for name, model in models.items():
        # Get evaluation metrics
        metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
        
        # Add to results with model name
        metrics['model_name'] = name
        
        # Add test/train ratio as indicator of overfitting
        metrics['test_train_ratio'] = metrics['test_rmse'] / metrics['train_rmse']
        
        results.append(metrics)
    
    return pd.DataFrame(results)


def plot_comparison(comparison_df: pd.DataFrame, metric: str = 'test_rmse') -> plt.Figure:
    """Plot a visual comparison of models based on a specific performance metric.
    
    This function creates a bar chart visualization comparing multiple models based on
    a selected performance metric from the comparison DataFrame. The visualization
    makes it easy to see which models perform best according to the chosen metric.
    
    Args:
        comparison_df (pd.DataFrame): DataFrame with model comparison results, typically
            generated by the compare_models function. Must contain columns 'model_name'
            and the specified metric.
        metric (str, optional): The metric column to use for comparison. Common options
            include 'test_rmse', 'test_r2', 'train_rmse', etc. Defaults to 'test_rmse'.
        
    Returns:
        plt.Figure: Matplotlib figure object containing the comparison bar chart.
            This can be displayed or saved as needed.
            
    Example:
        >>> comparison = compare_models(models, X_train, y_train, X_test, y_test)
        >>> fig = plot_comparison(comparison, metric='test_rmse')
        >>> plt.savefig('model_comparison.png', dpi=300)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.barplot(
        x='model_name',
        y=metric,
        data=comparison_df,
        ax=ax
    )
    
    ax.set_title(f'Model Comparison: {metric}')
    ax.set_xlabel('Model')
    ax.set_ylabel(metric)
    
    # Add values on top of bars
    for i, v in enumerate(comparison_df[metric]):
        ax.text(i, v + (v * 0.01), f"{v:.2f}", ha='center')
    
    plt.tight_layout()
    return fig


def plot_residuals(
    model: Model,
    X: pd.DataFrame,
    y: pd.Series,
    title: Optional[str] = None
) -> plt.Figure:
    """Plot residuals to assess model fit and identify potential issues.
    
    This function creates a comprehensive residual analysis plot with four subplots:
    1. Residuals vs. predicted values - to check for non-linearity and heteroscedasticity
    2. Distribution of residuals - to check for normality and variance
    3. Q-Q plot - to assess normality of residuals
    4. Residuals vs. order - to check for autocorrelation issues (assuming order matters)
    
    Residual analysis is crucial for validating regression model assumptions and
    identifying potential improvements. Patterns in residuals can reveal issues like
    non-linearity, heteroscedasticity, or autocorrelation that might need to be addressed.
    
    Args:
        model (Model): A trained model implementing the base Model interface
            with a predict() method.
        X (pd.DataFrame): Feature data for generating predictions and analyzing residuals.
        y (pd.Series): Actual target values to compare against predictions.
        title (Optional[str], optional): Title for the overall figure. If None, uses
            a default title based on the model type. Defaults to None.
            
    Returns:
        plt.Figure: Matplotlib figure object containing the four residual analysis
            subplots. This can be displayed or saved as needed.
            
    Example:
        >>> model = LinearRegressionModel().fit(X_train, y_train)
        >>> fig = plot_residuals(model, X_test, y_test, title="Ridge Model Residuals")
        >>> plt.savefig('ridge_residuals.png', dpi=300, bbox_inches='tight')
    """
    # Generate predictions and calculate residuals
    predictions = model.predict(X)
    residuals = y - predictions
    
    # Create a figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.tight_layout(pad=4)
    
    # Add a main title if provided
    if title is None:
        title = f"{model.model_type.upper()} Model Residual Analysis"
    fig.suptitle(title, fontsize=16, y=1.05)
    
    # 1. Residuals vs. Predicted Values (top left)
    axes[0, 0].scatter(predictions, residuals, alpha=0.5)
    axes[0, 0].axhline(y=0, color='r', linestyle='-')
    axes[0, 0].set_title('Residuals vs. Predicted Values')
    axes[0, 0].set_xlabel('Predicted Values')
    axes[0, 0].set_ylabel('Residuals')
    
    # Add a loess smoothing line to help identify patterns
    try:
        # Only add smoothing if there are sufficient data points
        if len(predictions) > 10:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            smoothed = lowess(residuals, predictions, frac=0.2)
            axes[0, 0].plot(smoothed[:, 0], smoothed[:, 1], 'r-', linewidth=2)
    except:
        # Skip smoothing if it fails
        pass
    
    # 2. Distribution of Residuals (top right)
    sns.histplot(residuals, kde=True, ax=axes[0, 1])
    axes[0, 1].axvline(x=0, color='r', linestyle='-')
    axes[0, 1].set_title('Distribution of Residuals')
    axes[0, 1].set_xlabel('Residual Value')
    axes[0, 1].set_ylabel('Frequency')
    
    # Add text with residual statistics
    stats_text = (f"Mean: {residuals.mean():.2f}\n"
                 f"St. Dev: {residuals.std():.2f}\n"
                 f"Skewness: {residuals.skew():.2f}")
    axes[0, 1].text(0.95, 0.95, stats_text, transform=axes[0, 1].transAxes,
                   ha='right', va='top', bbox=dict(facecolor='white', alpha=0.5))
    
    # 3. QQ Plot (bottom left)
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot of Residuals')
    
    # 4. Residuals vs. Order (bottom right)
    axes[1, 1].scatter(np.arange(len(residuals)), residuals, alpha=0.5)
    axes[1, 1].axhline(y=0, color='r', linestyle='-')
    axes[1, 1].set_title('Residuals vs. Order')
    axes[1, 1].set_xlabel('Observation Order')
    axes[1, 1].set_ylabel('Residuals')
    
    # Ensure the layout looks good
    plt.tight_layout()
    
    return fig
    # Generate predictions and calculate residuals
    predictions = model.predict(X)
    residuals = y - predictions
    
    # Create a figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.tight_layout(pad=4)
    
    # Add a main title if provided
    if title is None:
        title = f"{model.model_type.upper()} Model Residual Analysis"
    fig.suptitle(title, fontsize=16, y=1.05)
    
    # 1. Residuals vs. Predicted Values (top left)
    axes[0, 0].scatter(predictions, residuals, alpha=0.5)
    axes[0, 0].axhline(y=0, color='r', linestyle='-')
    axes[0, 0].set_title('Residuals vs. Predicted Values')
    axes[0, 0].set_xlabel('Predicted Values')
    axes[0, 0].set_ylabel('Residuals')
    
    # Add a loess smoothing line to help identify patterns
    try:
        # Only add smoothing if there are sufficient data points
        if len(predictions) > 10:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            smoothed = lowess(residuals, predictions, frac=0.2)
            axes[0, 0].plot(smoothed[:, 0], smoothed[:, 1], 'r-', linewidth=2)
    except:
        # Skip smoothing if it fails
        pass
    
    # 2. Distribution of Residuals (top right)
    sns.histplot(residuals, kde=True, ax=axes[0, 1])
    axes[0, 1].axvline(x=0, color='r', linestyle='-')
    axes[0, 1].set_title('Distribution of Residuals')
    axes[0, 1].set_xlabel('Residual Value')
    axes[0, 1].set_ylabel('Frequency')
    
    # Add text with residual statistics
    stats_text = (f"Mean: {residuals.mean():.2f}\n"
                 f"St. Dev: {residuals.std():.2f}\n"
                 f"Skewness: {residuals.skew():.2f}")
    axes[0, 1].text(0.95, 0.95, stats_text, transform=axes[0, 1].transAxes,
                   ha='right', va='top', bbox=dict(facecolor='white', alpha=0.5))
    
    # 3. QQ Plot (bottom left)
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot of Residuals')
    
    # 4. Residuals vs. Order (bottom right)
    axes[1, 1].scatter(np.arange(len(residuals)), residuals, alpha=0.5)
    axes[1, 1].axhline(y=0, color='r', linestyle='-')
    axes[1, 1].set_title('Residuals vs. Order')
    axes[1, 1].set_xlabel('Observation Order')
    axes[1, 1].set_ylabel('Residuals')
    
    # Ensure the layout looks good
    plt.tight_layout()
    
    return fig


def plot_prediction_error(
    model: Model,
    X: pd.DataFrame,
    y: pd.Series,
    title: Optional[str] = None
) -> plt.Figure:
    """Plot predicted vs actual values to visualize prediction accuracy.
    
    This function creates a joint plot showing the relationship between actual and
    predicted values, with marginal distributions for each. Points should ideally 
    fall along the y=x line, indicating perfect predictions.
    
    The plot helps identify:
    - Systematic over/under prediction across the value range
    - Range-specific accuracy issues (e.g., poor predictions for high-value properties)
    - Outliers where model predictions significantly differ from actual values
    
    Args:
        model (Model): Trained model with predict() method.
        X (pd.DataFrame): Feature data for which to generate predictions.
        y (pd.Series): Actual target values to compare with predictions.
        title (str, optional): Custom title for the plot. If None, uses a default title.
            Defaults to None.
        
    Returns:
        plt.Figure: Matplotlib figure object containing the joint plot of
            actual vs predicted values. This can be displayed or saved as needed.
            
    Example:
        >>> model = LinearRegressionModel().fit(X_train, y_train)
        >>> fig = plot_prediction_error(model, X_test, y_test)
        >>> plt.savefig('prediction_accuracy.png', dpi=300)
    """
    predictions = model.predict(X)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    g = sns.jointplot(
        x=y,
        y=predictions,
        alpha=0.5,
        height=8
    )
    
    # Draw a line of y=x
    x0, x1 = g.ax_joint.get_xlim()
    y0, y1 = g.ax_joint.get_ylim()
    lims = [max(x0, y0), min(x1, y1)]
    g.ax_joint.plot(lims, lims, '-r')
    
    if title is None:
        title = "Prediction Accuracy: Actual vs Predicted Values"
    
    g.fig.suptitle(title, y=1.05, fontsize=16)
    g.ax_joint.set_xlabel('True Values')
    g.ax_joint.set_ylabel('Predicted Values')
    
    plt.tight_layout()
    return g.fig
