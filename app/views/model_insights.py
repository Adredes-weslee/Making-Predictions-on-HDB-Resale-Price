"""Model insights component for the Streamlit application.

This module provides visualizations and explanations of the trained machine learning models
for HDB resale price prediction. It includes model performance metrics, feature importance
analysis, residual diagnostics, and a prediction explainer that breaks down how predictions
are made.

The page is divided into four main sections:
1. Model Performance - Shows metrics like R² and RMSE along with prediction vs actual plots
2. Feature Importance - Visualizes which features have the strongest impact on predictions
3. Residual Analysis - Examines the errors in predictions to assess model fit
4. Prediction Explainer - Breaks down how individual predictions are calculated

This component assumes trained models are available in the models directory.
"""
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import utility functions
from src.data.loader import get_data_paths, load_raw_data, load_train_test_data
from src.models.prediction import load_model
from src.models.evaluation import evaluate_model


def load_trained_model():
    """Load the trained model from the models directory.
    
    This function attempts to load the linear regression model from the models directory.
    It handles the case where the model file might not exist by catching the FileNotFoundError
    and displaying an appropriate error message to the user through Streamlit.
    
    Returns:
        object: The loaded model object if successful, None otherwise.
        
    Raises:
        FileNotFoundError: If the model file doesn't exist (handled internally).
    """
    base_dir = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    model_path = os.path.join(base_dir, "models", "linear_regression_model.pkl")
    
    try:
        model = load_model(model_path)
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at {model_path}. Please train the model first.")
        return None


def load_safe_data():
    """Load training data with error handling."""
    try:
        # Try to load processed data first
        data_paths = get_data_paths()
        processed_dir = data_paths["processed"]
        processed_file = os.path.join(processed_dir, "train_processed.csv")
        
        if os.path.exists(processed_file):
            df = load_raw_data(processed_file)
        else:
            df = load_raw_data(data_paths["train"])
            
        # Assume 'resale_price' is the target variable
        y = df["resale_price"]
        X = df.drop(columns=["resale_price"])
        
        # We don't use sklearn's train_test_split to avoid issues with dataset features
        return X, y
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None


def show_model_insights():
    """Display the model insights page of the application.
    
    This is the main entry point for the Model Insights page. The function:
    1. Loads the trained model
    2. Loads training and test data
    3. Creates a tabbed interface with different model analysis views
    4. Handles errors gracefully with informative messages
    
    The function uses nested try-except blocks to provide fallback behavior:
    - If the model can't be loaded, it shows an appropriate warning
    - If the training data can't be loaded, it attempts to show feature importance only
    - If both fail, it displays an error message
    
    Returns:
        None: This function renders Streamlit components directly.
    """
    st.title("HDB Resale Price Model Insights")
    
    # Load the trained model
    model = load_trained_model()
    if model is None:
        st.warning("Model could not be loaded. Insights are not available.")
        return
    
    # Try to load the data
    X_data, y_data = load_safe_data()
    
    if X_data is not None and y_data is not None:
        # Data loaded successfully - show full insights
        tabs = st.tabs(["Performance", "Feature Importance", "Residual Analysis", "Prediction Explainer"])
        
        with tabs[0]:
            show_model_performance(model, X_data, y_data)
        
        with tabs[1]:
            try:
                show_feature_importance(model, X_data)
            except Exception as e:
                st.warning(f"Error displaying feature importance with data: {str(e)}")
                st.info("Attempting to display feature importance without column names...")
                show_feature_importance(model, None)
        
        with tabs[2]:
            try:
                show_residual_analysis(model, X_data, y_data)
            except Exception as e:
                st.warning(f"Error displaying residual analysis: {str(e)}")
        
        with tabs[3]:
            try:
                show_prediction_explainer(model, X_data)
            except Exception as e:
                st.warning(f"Error displaying prediction explainer: {str(e)}")
    else:
        # Data loading failed - show limited insights
        st.warning("Could not load training data. Showing limited model insights.")
        
        tabs = st.tabs(["Feature Importance", "Model Information"])
        
        with tabs[0]:
            try:
                show_feature_importance(model, None)
            except Exception as e:
                st.error(f"Could not generate feature importance: {str(e)}")
        
        with tabs[1]:
            st.subheader("Model Information")
            st.write("#### Model Type")
            st.write(f"Model type: {type(model).__name__}")
            st.write("#### Model Parameters")
            if hasattr(model, 'get_params'):
                params = model.get_params()
                for param, value in params.items():
                    st.write(f"- **{param}**: {value}")
            else:
                st.write("Model parameters not available")


def show_model_performance(model, X_data, y_data):
    """Display model performance metrics and visualizations.
    
    This function evaluates the model on both training and testing data, then displays:
    1. Key performance metrics (R² and RMSE) for both train and test sets
    2. A scatter plot of predicted vs actual values
    3. A histogram of prediction errors (residuals)
    
    The performance metrics help assess both the model's accuracy (how well it fits the data)
    and its generalization (how well it performs on unseen data). The visualizations help
    identify patterns in the predictions and potential areas for improvement.
    
    Args:
        model (object): Trained machine learning model object with predict() method.
        X_train (pd.DataFrame): Training feature data.
        y_train (pd.Series): Training target values (actual resale prices).
        X_test (pd.DataFrame): Testing feature data.
        y_test (pd.Series): Testing target values (actual resale prices).
    
    Returns:
        None: This function renders Streamlit components directly.
    """
    st.header("Model Performance")
    
    # Split data for performance evaluation
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=123)
    
    # Evaluate the model
    try:
        metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
        
        # Display metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Train R²", f"{metrics['train_r2']:.4f}")
        
        with col2:
            st.metric("Test R²", f"{metrics['test_r2']:.4f}")
        
        with col3:
            st.metric("Train RMSE", f"${metrics['train_rmse']:.2f}")
        
        with col4:
            st.metric("Test RMSE", f"${metrics['test_rmse']:.2f}")
        
        # Show prediction vs actual plot
        st.subheader("Predicted vs Actual Values")
        
        # Make predictions on test data
        y_pred = model.predict(X_test)
        
        # Create scatter plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot predictions vs actual values
        ax.scatter(y_test, y_pred, alpha=0.5)
        
        # Add perfect prediction line
        max_val = max(y_test.max(), y_pred.max())
        min_val = min(y_test.min(), y_pred.min())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        # Set labels and title
        ax.set_xlabel('Actual Price')
        ax.set_ylabel('Predicted Price')
        ax.set_title('Predicted vs Actual HDB Resale Prices')
        
        # Format axis labels
        ax.ticklabel_format(style='plain', axis='both')
        
        st.pyplot(fig)
        
        # Display error distribution
        st.subheader("Error Distribution")
        
        # Calculate errors
        errors = y_test - y_pred
        
        # Create histogram of errors
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.histplot(errors, kde=True, ax=ax)
        
        ax.set_xlabel('Prediction Error')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Prediction Errors')
        
        # Add a vertical line at zero
        ax.axvline(x=0, color='r', linestyle='--')
        
        # Add statistics to plot
        error_mean = errors.mean()
        error_std = errors.std()
        
        ax.text(
            0.95, 0.95,
            f'Mean Error: ${error_mean:.2f}\nStd Dev: ${error_std:.2f}',
            transform=ax.transAxes,
            ha='right',
            va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error evaluating model: {str(e)}")


def show_feature_importance(model, X_train):
    """Display feature importance visualizations for the trained model.
    
    This function creates and displays visualizations showing which features have the
    most impact on the model's predictions. It supports both tree-based models (using
    feature_importances_) and linear models (using coef_). The function automatically
    detects the model type and generates appropriate visualizations.
    
    For tree-based models, a bar chart of feature importance values is shown.
    For linear models, a bar chart of coefficients is shown with color-coding to
    indicate positive (green) and negative (red) relationships.
    
    The function includes a fallback mechanism to handle cases where X_train is not 
    provided, using generic feature names instead of actual column names.
    
    Args:
        model (object): Trained machine learning model with feature_importances_
            or coef_ attribute.
        X_train (pd.DataFrame or None): Training features dataframe. If None,
            generic feature names will be used.
            
    Returns:
        None: This function renders Streamlit components directly.
        
    Raises:
        AttributeError: If the model doesn't have feature_importances_ or coef_.
            This is handled internally with an appropriate error message.
    """
    st.header("Feature Importance")
    
    try:
        # Check if model has feature importance or coefficients
        if hasattr(model, 'feature_importances_'):
            # For tree-based models
            if X_train is not None:
                # Get feature names and importances
                feature_names = X_train.columns
                importances = model.feature_importances_
            else:
                # Fallback if no training data available
                feature_names = [f"Feature {i}" for i in range(len(model.feature_importances_))]
                importances = model.feature_importances_
            
            # Create feature importance dataframe
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            # Plot feature importance
            fig, ax = plt.subplots(figsize=(10, max(6, min(20, len(feature_names) * 0.3))))
            
            sns.barplot(x='Importance', y='Feature', data=importance_df.head(20), ax=ax)
            
            ax.set_title('Feature Importance')
            ax.set_xlabel('Importance')
            ax.set_ylabel('Feature')
            
            st.pyplot(fig)
            
        elif hasattr(model, 'coef_'):
            # For linear models
            if X_train is not None:
                # Get feature names and coefficients
                feature_names = X_train.columns
                coefficients = model.coef_
            else:
                # Fallback if no training data available
                feature_names = [f"Feature {i}" for i in range(len(model.coef_))]
                coefficients = model.coef_
            
            # Create coefficients dataframe
            coef_df = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': coefficients
            })
            
            # Sort by absolute coefficient value
            coef_df['AbsCoef'] = coef_df['Coefficient'].abs()
            coef_df = coef_df.sort_values('AbsCoef', ascending=False).drop(columns='AbsCoef')
            
            # Plot top coefficients
            fig, ax = plt.subplots(figsize=(10, max(6, min(20, len(feature_names) * 0.3))))
            
            # Color based on sign of coefficient
            colors = ['green' if c > 0 else 'red' for c in coef_df.head(20)['Coefficient']]
            
            # Create the bar plot
            bars = sns.barplot(
                x='Coefficient',
                y='Feature',
                data=coef_df.head(20),
                ax=ax,
                palette=colors
            )
            
            ax.set_title('Feature Coefficients (Green = Positive Impact, Red = Negative Impact)')
            ax.set_xlabel('Coefficient Value')
            ax.set_ylabel('Feature')
            
            st.pyplot(fig)
            
            # Display information about top features
            st.write("### Top Features Explanation")
            st.write("""
            The chart above shows the features with the strongest impact on the predicted resale price.
            - **Green bars** represent features that increase the price when their values increase
            - **Red bars** represent features that decrease the price when their values increase
            
            The length of each bar indicates the magnitude of the effect on the final price.
            """)
        else:
            st.warning("The model doesn't provide direct feature importance or coefficients.")
    except Exception as e:
        st.error(f"Error calculating feature importance: {str(e)}")


def show_residual_analysis(model, X_data, y_data):
    """Display residual analysis visualizations to assess model fit quality.
    
    This function creates and displays a comprehensive set of residual analysis
    visualizations to help understand the model's performance and potential issues:
    
    1. Residuals vs Predicted Values scatter plot - Shows if residuals are randomly 
       distributed (good) or have patterns (bad)
    2. Q-Q Plot - Assesses if residuals follow a normal distribution
    3. Percentage Error Distribution - Shows the distribution of errors as percentages
    4. Tables of largest over/under predictions - Identifies outliers or systematic errors
    5. Error statistics summary - Provides quantitative error measures
    
    These visualizations help identify issues such as:
    - Heteroscedasticity (non-constant variance)
    - Non-normality of residuals
    - Outliers and influential points
    - Systematic over/under prediction for certain cases
    
    Args:
        model (object): Trained machine learning model with predict() method.
        X_test (pd.DataFrame): Testing feature data.
        y_test (pd.Series): Testing target values (actual resale prices).
    
    Returns:
        None: This function renders Streamlit components directly.
    """
    st.header("Residual Analysis")
    
    # Make predictions on the data
    y_pred = model.predict(X_data)
    residuals = y_data - y_pred
    
    # Create visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Residual plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_pred, residuals, alpha=0.5)
        ax.axhline(y=0, color='r', linestyle='-')
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Residuals')
        ax.set_title('Residual Plot')
        ax.ticklabel_format(style='plain', axis='x')
        st.pyplot(fig)
    
    with col2:
        # Q-Q plot for residuals
        from scipy import stats
        
        fig, ax = plt.subplots(figsize=(10, 6))
        stats.probplot(residuals, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot of Residuals')
        st.pyplot(fig)
    
    # Distribution of residuals
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(residuals, kde=True, ax=ax)
    ax.set_xlabel('Residuals')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Residuals')
    st.pyplot(fig)
    
    # Display statistics
    residual_stats = {
        "Mean": residuals.mean(),
        "Standard Deviation": residuals.std(),
        "Minimum": residuals.min(),
        "Maximum": residuals.max(),
        "Skewness": residuals.skew()
    }
    
    st.write("### Residual Statistics")
    
    stats_df = pd.DataFrame({
        "Value": [f"${val:.2f}" if i < 4 else f"{val:.4f}" for i, val in enumerate(residual_stats.values())]
    }, index=residual_stats.keys())
    
    st.dataframe(stats_df)


def show_prediction_explainer(model, X_data):
    """Display an interactive explainer showing how predictions are calculated.
    
    This function creates an interpretable breakdown of how the model makes predictions:
    1. Allows user to select a sample property from the training data
    2. Shows a visualization of how each feature contributes to the final prediction
    3. Breaks down the mathematical calculation from feature values to final price
    4. Highlights which features have the largest positive and negative impacts
    
    The explainer only works with linear models that have a coefficients (coef_)
    attribute, as it needs to multiply feature values by their coefficients to 
    show contributions. For non-linear models, a warning is displayed instead.
    
    Args:
        model (object): Trained machine learning model with coef_ attribute
            (typically a linear model).
        X_train (pd.DataFrame): Training feature data used to select sample properties.
    
    Returns:
        None: This function renders Streamlit components directly.
        
    Notes:
        The explainer breaks down the linear model calculation: 
        prediction = intercept + (feature1 * coef1) + (feature2 * coef2) + ...
        
        The charts and tables help users understand which features increase the
        price (green bars) and which decrease it (red bars), with the magnitude
        of each contribution shown.
    """
    st.header("Prediction Explainer")
    
    st.write("""
    This tool helps you understand how the model calculates predictions for individual properties.
    Select a sample property from the data or enter custom values to see how different features
    contribute to the final prediction.
    """)
    
    # Only continue if we have the appropriate model and data
    if X_data is not None and hasattr(model, 'coef_') and hasattr(model, 'intercept_'):
        # Allow user to select a sample or enter custom values
        option = st.radio(
            "Choose an option:",
            ["Select a sample property", "Enter custom values"]
        )
        
        if option == "Select a sample property":
            # Let the user select a sample from the data
            sample_index = st.slider("Select a sample property:", 0, len(X_data) - 1, 0)
            sample = X_data.iloc[sample_index:sample_index+1]
        else:
            # For simplicity, we'll just create a sample with median values
            # In a real app, you'd have input widgets for each feature
            st.warning("Custom value input not implemented in this demo. Using median values.")
            sample = pd.DataFrame({col: [X_data[col].median()] for col in X_data.columns})
        
        # Make prediction for the sample
        prediction = model.predict(sample)[0]
        
        # Get the intercept and coefficients from the model
        intercept = model.intercept_
        coefficients = model.coef_
        
        # Calculate feature contributions
        contributions = {}
        contributions['intercept'] = intercept
        
        for i, col in enumerate(X_data.columns):
            contributions[col] = coefficients[i] * sample[col].values[0]
        
        # Convert to a dataframe for display
        contrib_df = pd.DataFrame({
            'Feature': list(contributions.keys()),
            'Value': [sample[col].values[0] if col != 'intercept' else 1 for col in contributions.keys()],
            'Coefficient': [coefficients[i] if col != 'intercept' else intercept for i, col in enumerate(contributions.keys()) if col != 'intercept'],
            'Contribution': list(contributions.values())
        })
        
        # Sort by absolute contribution
        contrib_df['AbsContribution'] = contrib_df['Contribution'].abs()
        contrib_df = contrib_df.sort_values('AbsContribution', ascending=False).drop(columns='AbsContribution')
        
        # Display the prediction
        st.subheader(f"Predicted Price: ${prediction:,.2f}")
        
        # Plot the contributions
        fig, ax = plt.subplots(figsize=(10, 12))
        
        # Get top contributors
        top_contribs = contrib_df.head(15)
        
        # Create waterfall chart colors
        colors = ['green' if c > 0 else 'red' for c in top_contribs['Contribution']]
        
        # Create the bar plot
        bars = ax.barh(top_contribs['Feature'], top_contribs['Contribution'], color=colors)
        
        ax.set_title('Top Feature Contributions to Prediction')
        ax.set_xlabel('Contribution to Price ($)')
        ax.ticklabel_format(style='plain', axis='x')
        
        st.pyplot(fig)
        
        # Display the contribution table
        st.write("### Feature Contributions")
        formatted_df = contrib_df.copy()
        formatted_df['Contribution'] = formatted_df['Contribution'].apply(lambda x: f"${x:,.2f}")
        st.dataframe(formatted_df.head(20), use_container_width=True)
        
        # Explanation
        st.write("""
        ### How to Interpret:
        - The base price (intercept) is the starting point for all predictions
        - Each feature adds or subtracts from this base price
        - Green bars represent features that increase the price
        - Red bars represent features that decrease the price
        - The final prediction is the sum of the base price and all feature contributions
        """)
    else:
        st.warning("""
        This functionality is only available for linear models that provide coefficients.
        The current model doesn't support detailed explainability.
        """)
