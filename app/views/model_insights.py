"""Model Insights view for the Streamlit application.

This module provides visualizations and analysis of model performance and feature importance
to help users understand how the prediction models work and what factors influence resale prices.
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px  # For histograms
from pathlib import Path
from components.visualizations import (
    plot_model_performance_comparison,
    plot_feature_importance
)

def show_model_insights():
    """Display model performance and feature importance visualizations."""
    # Header
    st.markdown("<h1 class='main-header'>Model Performance</h1>", unsafe_allow_html=True)
    
    # Ensure models are loaded
    if 'models' not in st.session_state:
        st.error("Models not loaded. Please refresh the application.")
        return
        
    models_dict = st.session_state['models']
    
    # Check if we have any models loaded
    model_types = ['linear', 'ridge', 'lasso']
    available_models = [m for m in model_types if f"{m}_model" in models_dict and models_dict[f"{m}_model"] is not None]
    
    if not available_models:
        st.error("No models available for analysis.")
        return
    
    # Introduction text
    st.markdown("""
    This section provides insights into how our machine learning models perform and which features
    have the most influence on HDB resale prices. Understanding these factors can help buyers and 
    sellers make more informed decisions about property transactions.
    """)
    
    # Model performance comparison
    st.markdown("## Model Performance Comparison")
    
    perf_fig = plot_model_performance_comparison(models_dict)
    if perf_fig:
        st.plotly_chart(perf_fig, use_container_width=True)
        
        # Add explanation
        st.markdown("""
        **What these metrics mean:**
        - **R² Score**: Indicates how well the model explains the variance in prices (higher is better, max is 1.0)
        - **RMSE (Root Mean Square Error)**: Average prediction error in dollars (lower is better)
        
        **Comparing training vs. testing metrics:**
        - Similar performance between training and testing data suggests good model generalization
        - Much better performance on training than testing suggests overfitting
        """)
    else:
        st.info("Model performance comparison not available")
    
    # Feature importance analysis
    st.markdown("## Feature Importance Analysis")
    
    # Select model for feature importance
    selected_model = st.selectbox(
        "Select Model for Feature Analysis",
        options=available_models,
        index=0 if 'ridge' in available_models else 0,
        format_func=lambda x: x.capitalize()
    )
    
    # Plot feature importance
    imp_fig = plot_feature_importance(models_dict, selected_model)
    if imp_fig:
        st.plotly_chart(imp_fig, use_container_width=True)
        
        # Add explanation
        st.markdown("""
        **Understanding Feature Importance:**
        
        The chart above shows which features have the strongest influence on price predictions.
        Features with higher coefficients (longer bars) have more impact on the predicted price.
        
        **Key insights:**
        - Location features (towns) are typically among the most important factors
        - Physical attributes like floor area and storey range are also significant
        - Remaining lease duration has a notable impact on resale value
        """)
    else:
        st.info("Feature importance visualization not available for the selected model")
    
    # Model evaluation details
    st.markdown("## Detailed Model Metrics")
    
    # Create metrics table
    metrics_data = []
    for model_type in model_types:
        if f"{model_type}_metrics" in models_dict and models_dict[f"{model_type}_metrics"] is not None:
            metrics = models_dict[f"{model_type}_metrics"]
            metrics_data.append({
                "Model": model_type.capitalize(),
                "Training R²": f"{metrics.get('train_r2', 'N/A'):.4f}",
                "Testing R²": f"{metrics.get('test_r2', 'N/A'):.4f}",
                "Training RMSE": f"${metrics.get('train_rmse', 'N/A'):,.2f}",
                "Testing RMSE": f"${metrics.get('test_rmse', 'N/A'):,.2f}"
            })
    
    if metrics_data:
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True)
        
        # Find best model
        best_model_idx = max(range(len(metrics_data)), 
                           key=lambda i: float(metrics_data[i]["Testing R²"].replace("N/A", "0")))
        best_model = metrics_data[best_model_idx]["Model"]
        
        st.success(f"**Best Performing Model:** {best_model}")
        
    else:
        st.info("Detailed metrics not available")
    
    st.markdown("## Model Coefficient Analysis")
        
    # Compare model coefficients to diagnose prediction differences
    def compare_model_coefficients(models_dict):
        """Compare coefficients between models to diagnose prediction differences."""
        for model_type in ['linear', 'ridge', 'lasso']:
            # Use the correct key that was set in main.py
            regressor_key = f"{model_type}_regressor"
            if regressor_key in models_dict and models_dict[regressor_key] is not None:
                model = models_dict[regressor_key]
                if hasattr(model, 'coef_'):
                    # Get coefficient stats
                    coefs = model.coef_
                    non_zero = np.sum(coefs != 0)
                    total = len(coefs)
                    largest_coef = np.max(np.abs(coefs))
                    
                    st.subheader(f"{model_type.capitalize()} Model")
                    
                    # Create columns for metrics
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Non-zero Coefficients", f"{non_zero}/{total}")
                    col2.metric("Sparsity", f"{(total-non_zero)/total:.1%}")
                    col3.metric("Largest Coefficient", f"{largest_coef:.2f}")
                    
                    # Create histogram of coefficients
                    fig = px.histogram(
                        x=coefs, 
                        nbins=50,
                        title=f"{model_type.capitalize()} Coefficient Distribution",
                        labels={"x": "Coefficient Value", "y": "Count"}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"{model_type} model has no coefficients available")
            else:
                st.warning(f"{model_type} regressor not found in models")
        
    # Call the function
    compare_model_coefficients(models_dict)

    # Add explanation
    st.markdown("""
    **What this analysis tells you:**
    - **Non-zero coefficients:** Models with more non-zero coefficients use more features for prediction
    - **Sparsity:** Lasso typically has higher sparsity (more zeros) due to L1 regularization
    - **Distribution:** Shows how coefficient values are distributed (many small vs few large values)

    Different coefficient patterns explain why models might give different predictions for the same input.
""")
    
    # Calculate correlation between model coefficients
    if all(f"{m}_regressor" in models_dict and models_dict[f"{m}_regressor"] is not None for m in ['ridge', 'linear', 'lasso']):
        st.markdown("### Coefficient Correlation Between Models")
        
        # Get coefficients for all models
        ridge_coefs = models_dict["ridge_regressor"].coef_
        linear_coefs = models_dict["linear_regressor"].coef_
        lasso_coefs = models_dict["lasso_regressor"].coef_
        
        # Create correlation matrix
        corr_matrix = np.corrcoef([ridge_coefs, linear_coefs, lasso_coefs])
        corr_df = pd.DataFrame(
            corr_matrix, 
            columns=["Ridge", "Linear", "Lasso"],
            index=["Ridge", "Linear", "Lasso"]
        )
        
        # Format to percentage
        formatted_corr = corr_df.applymap(lambda x: f"{x:.1%}")
        st.dataframe(formatted_corr)
        
        st.markdown("""
        **Coefficient correlation** shows how similar the feature importance patterns are between models.
        Low correlation between Lasso and other models explains prediction differences!
        """)
    else:
        st.warning("Not all model regressors are available for correlation analysis")
    
    # Model methodology
    st.markdown("## Methodology")
    st.markdown("""
    ### Data Preparation
    
    The models were trained on HDB resale transactions from data.gov.sg, with:
    - Feature engineering to create meaningful predictors
    - Handling of categorical variables through encoding
    - Removal of outliers to improve model robustness
    - Normalization of numerical features
    
    ### Model Training
    
    We trained and evaluated several regression models:
    
    1. **Linear Regression**: Basic model that assumes linear relationships
    2. **Ridge Regression**: Adds regularization to prevent overfitting
    3. **Lasso Regression**: Adds regularization and performs feature selection
    
    ### Validation
    
    Models were validated using:
    - Train-test split (80% training, 20% testing)
    - Standard regression metrics (R², RMSE)
    - Residual analysis to check for bias
    """)