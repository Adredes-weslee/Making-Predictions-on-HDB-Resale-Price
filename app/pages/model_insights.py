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
    
    try:
        # Load the trained model
        model = load_trained_model()
        if model is None:
            st.warning("Model could not be loaded. Some insights may not be available.")
            return
        
        try:
            # Load the train/test data
            X_train, X_test, y_train, y_test = load_train_test_data()
            
            # Create tabs for different insights
            tabs = st.tabs(["Performance", "Feature Importance", "Residual Analysis", "Prediction Explainer"])
            
            with tabs[0]:
                show_model_performance(model, X_train, y_train, X_test, y_test)
                
            with tabs[1]:
                show_feature_importance(model, X_train)
                
            with tabs[2]:
                show_residual_analysis(model, X_test, y_test)
                
            with tabs[3]:
                show_prediction_explainer(model, X_train)
                
        except Exception as e:
            st.warning(f"Could not load training data: {str(e)}")
            st.write("Showing limited model insights based on model structure only.")
            
            # Try to show feature importance at minimum
            show_feature_importance(model, None)
    
    except Exception as e:
        st.error(f"Error loading model or data: {str(e)}")
        st.write("Please ensure that the model has been trained and data is available.")
        st.write("Run the data processing and model training scripts to prepare the necessary files.")


def show_model_performance(model, X_train, y_train, X_test, y_test):
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
    
    # Evaluate the model
    metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
    
    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Training R²", f"{metrics['train_r2']:.4f}")
    
    with col2:
        st.metric("Testing R²", f"{metrics['test_r2']:.4f}")
    
    with col3:
        st.metric("Training RMSE", f"${metrics['train_rmse']:,.2f}")
    
    with col4:
        st.metric("Testing RMSE", f"${metrics['test_rmse']:,.2f}")
    
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
            fig, ax = plt.subplots(figsize=(10, max(6, len(feature_names) * 0.3)))
            
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
            fig, ax = plt.subplots(figsize=(10, max(6, len(feature_names) * 0.3)))
            
            # Color based on sign of coefficient
            colors = ['green' if c > 0 else 'red' for c in coef_df.head(20)['Coefficient']]
            
            bars = sns.barplot(x='Coefficient', y='Feature', data=coef_df.head(20), palette=colors, ax=ax)
            
            ax.set_title('Feature Coefficients')
            ax.set_xlabel('Coefficient Value')
            ax.set_ylabel('Feature')
            
            # Add a vertical line at zero
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            
            st.pyplot(fig)
            
            # Display interpretation
            st.subheader("Interpretation")
            
            st.write("""
            The chart above shows the coefficients of the linear regression model. 
            These coefficients represent how much the price changes when the feature increases by one unit.
            
            - **Green bars (positive coefficients)** indicate features that tend to increase the price when their value increases.
            - **Red bars (negative coefficients)** indicate features that tend to decrease the price when their value increases.
            
            The longer the bar, the more impact the feature has on the price prediction.
            """)
            
        else:
            st.warning("The model doesn't provide direct feature importance or coefficients.")
    except Exception as e:
        st.error(f"Error displaying feature importance: {str(e)}")


def show_residual_analysis(model, X_test, y_test):
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
    
    # Make predictions on test data
    y_pred = model.predict(X_test)
    
    # Calculate residuals
    residuals = y_test - y_pred
    
    # Create DataFrame with predictions and residuals
    results_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred,
        'Residual': residuals,
        'AbsResidual': np.abs(residuals),
        'PercentError': np.abs(residuals) / y_test * 100
    })
    
    # Residuals vs Predicted plot
    st.subheader("Residuals vs Predicted Values")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.scatter(y_pred, residuals, alpha=0.5)
    ax.axhline(y=0, color='r', linestyle='--')
    
    ax.set_xlabel('Predicted Price')
    ax.set_ylabel('Residual')
    ax.set_title('Residuals vs Predicted Values')
    
    # Format x-axis labels
    ax.ticklabel_format(style='plain', axis='x')
    
    st.pyplot(fig)
    
    # Q-Q plot for residuals
    st.subheader("Q-Q Plot of Residuals")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=ax)
    
    ax.set_title('Q-Q Plot of Residuals')
    
    st.pyplot(fig)
    
    # Histogram of percent errors
    st.subheader("Distribution of Percentage Errors")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.histplot(results_df['PercentError'], kde=True, ax=ax)
    
    ax.set_xlabel('Percentage Error')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Percentage Errors')
    
    st.pyplot(fig)
    
    # Top over/under predictions
    st.subheader("Largest Prediction Errors")
    
    # Select relevant columns for X_test
    if 'town' in X_test.columns and 'flat_type' in X_test.columns:
        # Combine results with original features
        results_df['Town'] = X_test['town'].reset_index(drop=True)
        results_df['Flat Type'] = X_test['flat_type'].reset_index(drop=True)
        
        if 'floor_area_sqm' in X_test.columns:
            results_df['Floor Area'] = X_test['floor_area_sqm'].reset_index(drop=True)
        
        display_cols = ['Town', 'Flat Type', 'Actual', 'Predicted', 'Residual', 'PercentError']
        
        # Find top under-predictions
        st.write("#### Top Under-Predictions")
        st.dataframe(results_df.sort_values('Residual', ascending=False).head(10)[display_cols], use_container_width=True)
        
        # Find top over-predictions
        st.write("#### Top Over-Predictions")
        st.dataframe(results_df.sort_values('Residual').head(10)[display_cols], use_container_width=True)
    else:
        # Just show the errors without features
        display_cols = ['Actual', 'Predicted', 'Residual', 'PercentError']
        
        # Find top under-predictions
        st.write("#### Top Under-Predictions")
        st.dataframe(results_df.sort_values('Residual', ascending=False).head(10)[display_cols], use_container_width=True)
        
        # Find top over-predictions
        st.write("#### Top Over-Predictions")
        st.dataframe(results_df.sort_values('Residual').head(10)[display_cols], use_container_width=True)
    
    # Summary statistics of errors
    st.subheader("Error Statistics")
    
    error_stats = {
        "Mean Absolute Error": results_df['AbsResidual'].mean(),
        "Median Absolute Error": results_df['AbsResidual'].median(),
        "Mean Percentage Error": results_df['PercentError'].mean(),
        "Median Percentage Error": results_df['PercentError'].median(),
        "90th Percentile Error": np.percentile(results_df['PercentError'], 90),
        "95th Percentile Error": np.percentile(results_df['PercentError'], 95)
    }
    
    # Convert to DataFrame for display
    stats_df = pd.DataFrame({"Value": error_stats}).T
    st.dataframe(stats_df, use_container_width=True)


def show_prediction_explainer(model, X_train):
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
    st.header("Prediction Explanation")
    
    st.write("""
    This section helps you understand how the model makes predictions by breaking down 
    the contribution of each feature to the final price. Select sample properties below 
    to see how different features impact the predicted price.
    """)
    
    try:
        # Check if model has coefficients (linear model)
        if hasattr(model, 'coef_') and X_train is not None:
            # Get intercept (base price)
            intercept = model.intercept_
            
            # Get coefficients
            feature_names = X_train.columns
            coefficients = model.coef_
            
            # Create feature importance dataframe
            model_factors = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': coefficients
            })
            
            # Select random sample properties for explanation
            sample_indices = np.random.choice(len(X_train), min(5, len(X_train)), replace=False)
            sample_properties = X_train.iloc[sample_indices].reset_index(drop=True)
            
            # Let user select which sample to explain
            selected_sample = st.selectbox(
                "Select a sample property to explain:",
                options=list(range(len(sample_properties))), 
                format_func=lambda x: f"Property {x+1}"
            )
            
            # Display selected sample
            selected_property = sample_properties.iloc[selected_sample]
            
            # Display property details
            st.subheader(f"Property {selected_sample+1} Details")
            
            # Check for key features to display
            key_features = ['town', 'flat_type', 'storey_range', 'floor_area_sqm']
            display_features = [f for f in key_features if f in selected_property.index]
            
            # Create columns for key features
            cols = st.columns(min(4, len(display_features)))
            for i, feature in enumerate(display_features):
                if i < len(cols):
                    cols[i].metric(
                        feature.replace('_', ' ').title(),
                        selected_property[feature]
                    )
            
            # Calculate feature contributions
            contributions = []
            
            # Add intercept (base price)
            contributions.append({
                'Feature': 'Base Price',
                'Value': 'N/A',
                'Contribution': intercept,
                'Coefficient': 'N/A'
            })
            
            # Calculate contribution for each feature
            for feature in feature_names:
                feature_value = selected_property[feature]
                feature_coef = model_factors[model_factors['Feature'] == feature]['Coefficient'].values[0]
                contribution = feature_value * feature_coef
                
                contributions.append({
                    'Feature': feature,
                    'Value': feature_value,
                    'Contribution': contribution,
                    'Coefficient': feature_coef
                })
            
            # Convert to DataFrame
            contributions_df = pd.DataFrame(contributions)
            
            # Sort by absolute contribution
            contributions_df['AbsContribution'] = contributions_df['Contribution'].abs()
            contributions_df = contributions_df.sort_values('AbsContribution', ascending=False)
            
            # Calculate total prediction
            prediction = contributions_df['Contribution'].sum()
            
            # Display prediction
            st.metric("Predicted Price", f"${prediction:,.2f}")
            
            # Visualize top feature contributions
            st.subheader("Feature Contributions")
            
            # Get top contributing features
            top_contributions = contributions_df.head(10).copy()
            
            # Plot horizontal bar chart
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Color based on sign of contribution
            colors = ['green' if c > 0 else 'red' for c in top_contributions['Contribution']]
            
            sns.barplot(
                x='Contribution', 
                y='Feature', 
                data=top_contributions, 
                palette=colors,
                ax=ax
            )
            
            ax.set_title('Top Feature Contributions to Prediction')
            ax.set_xlabel('Contribution to Price (SGD)')
            ax.set_ylabel('Feature')
            
            # Add a vertical line at zero
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            
            # Format x-axis labels
            ax.ticklabel_format(style='plain', axis='x')
            
            st.pyplot(fig)
            
            # Display full contribution breakdown
            st.subheader("Detailed Contribution Breakdown")
            
            # Format DataFrame for display
            display_df = contributions_df[['Feature', 'Value', 'Coefficient', 'Contribution']].copy()
            
            # Format the contribution and coefficient columns
            display_df['Contribution'] = display_df['Contribution'].apply(lambda x: f"${x:,.2f}" if isinstance(x, (int, float)) else x)
            display_df['Coefficient'] = display_df['Coefficient'].apply(lambda x: f"{x:,.4f}" if isinstance(x, (int, float)) else x)
            
            st.dataframe(display_df.drop(columns='AbsContribution'), use_container_width=True)
            
        else:
            st.warning("Prediction explanation is only available for linear models with coefficients.")
            
    except Exception as e:
        st.error(f"Error in prediction explainer: {str(e)}")
        st.warning("Could not generate prediction explanation.")
