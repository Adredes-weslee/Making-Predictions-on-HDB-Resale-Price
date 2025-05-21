"""Reusable visualization components.

This module contains functions that create various types of visualizations used
throughout the application. Each function follows a similar pattern of accepting
data and configuration parameters, and returning a visualization object that can
be displayed in the Streamlit UI.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from pathlib import Path

# Set style for matplotlib
plt.style.use('ggplot')
sns.set_style("whitegrid")

def plot_model_performance_comparison(models_dict):
    """Create a bar chart comparing model performance metrics.
    
    Args:
        models_dict: Dictionary containing model metrics
        
    Returns:
        plotly.graph_objects.Figure: Interactive model comparison chart
    """
    try:
        # Extract metrics
        model_names = []
        train_r2 = []
        test_r2 = []
        train_rmse = []
        test_rmse = []
        
        for model_type in ['linear', 'ridge', 'lasso']:
            if f"{model_type}_metrics" in models_dict and models_dict[f"{model_type}_metrics"] is not None:
                metrics = models_dict[f"{model_type}_metrics"]
                model_names.append(model_type.capitalize())
                train_r2.append(metrics.get('train_r2', 0))
                test_r2.append(metrics.get('test_r2', 0))
                train_rmse.append(metrics.get('train_rmse', 0))
                test_rmse.append(metrics.get('test_rmse', 0))
        
        if not model_names:
            return None
            
        # Create figure with two subplots
        fig = make_subplots(
            rows=1, cols=2, 
            subplot_titles=("R² Score (higher is better)", "RMSE (lower is better)"),
            specs=[[{"type": "bar"}, {"type": "bar"}]],
            horizontal_spacing=0.1
        )
        
        # Add R² bars
        fig.add_trace(
            go.Bar(name='Training R²', x=model_names, y=train_r2, marker_color='#1E88E5'),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(name='Testing R²', x=model_names, y=test_r2, marker_color='#5E35B1'),
            row=1, col=1
        )
        
        # Add RMSE bars
        fig.add_trace(
            go.Bar(name='Training RMSE', x=model_names, y=train_rmse, marker_color='#43A047'),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(name='Testing RMSE', x=model_names, y=test_rmse, marker_color='#FB8C00'),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Model Performance Comparison",
            height=500,
            legend_title_text="Metric Type",
            barmode='group',
            template='plotly_white'
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating model performance chart: {str(e)}")
        return None

def plot_price_distribution(df):
    """Plot the distribution of resale prices.
    
    Args:
        df: DataFrame containing resale price data
        
    Returns:
        plotly.graph_objects.Figure: Histogram of resale prices
    """
    try:
        fig = px.histogram(
            df, 
            x="resale_price", 
            nbins=50,
            title="Distribution of HDB Resale Prices",
            labels={"resale_price": "Resale Price (SGD)"}
        )
        fig.update_layout(
            xaxis_title="Resale Price (SGD)",
            yaxis_title="Count",
            bargap=0.1
        )
        return fig
    except Exception as e:
        st.error(f"Error creating price distribution chart: {str(e)}")
        return None

def plot_price_trends(df):
    """Plot price trends over time.
    
    Args:
        df: DataFrame with date and price columns
        
    Returns:
        plotly.graph_objects.Figure: Line chart of price trends
    """
    try:
        # Ensure df has datetime format
        if 'year_month' in df.columns:
            df_trend = df.copy()
            
            # Group by year_month and calculate average price
            df_trend = df_trend.groupby('year_month')['resale_price'].mean().reset_index()
            
            fig = px.line(
                df_trend, 
                x="year_month", 
                y="resale_price",
                title="Average HDB Resale Price Trends",
                labels={
                    "year_month": "Year-Month",
                    "resale_price": "Average Resale Price (SGD)"
                }
            )
            fig.update_layout(
                xaxis_title="Year-Month",
                yaxis_title="Average Resale Price (SGD)",
                hovermode="x unified"
            )
            return fig
        else:
            return None
    except Exception as e:
        st.error(f"Error creating price trends chart: {str(e)}")
        return None

def plot_feature_importance(models_dict, selected_model='ridge'):
    """Plot feature importance for a selected model.
    
    Args:
        models_dict: Dictionary containing models and feature info
        selected_model: Type of model to display feature importance for
        
    Returns:
        plotly.graph_objects.Figure: Feature importance bar chart
    """
    try:
        if f"{selected_model}_model" not in models_dict or models_dict[f"{selected_model}_model"] is None:
            return None
            
        model = models_dict[f"{selected_model}_model"]
        
        # Get feature names and coefficients
        try:
            # For pipeline models, extract from named steps
            if hasattr(model, 'named_steps') and 'model' in model.named_steps:
                coefficients = model.named_steps['model'].coef_
                
                # Get feature names from features json if available
                if f"{selected_model}_features" in models_dict and models_dict[f"{selected_model}_features"] is not None:
                    feature_info = models_dict[f"{selected_model}_features"]
                    if 'transformed_features' in feature_info:
                        feature_names = feature_info['transformed_features']
                    else:
                        feature_names = [f"feature_{i}" for i in range(len(coefficients))]
                else:
                    feature_names = [f"feature_{i}" for i in range(len(coefficients))]
            else:
                # For standard models, use coef_ attribute directly
                coefficients = model.coef_
                feature_names = [f"feature_{i}" for i in range(len(coefficients))]
        
        except Exception:
            # Fallback if model structure isn't as expected
            st.warning("Could not extract feature importance from model")
            return None
        
        # Create DataFrame for the chart
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': np.abs(coefficients)
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Coefficient', ascending=False).head(20)
        
        # Create bar chart
        fig = px.bar(
            importance_df, 
            y='Feature', 
            x='Coefficient',
            orientation='h',
            title=f"Top 20 Important Features - {selected_model.capitalize()} Model",
            labels={"Coefficient": "Absolute Coefficient Value", "Feature": "Feature Name"}
        )
        
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            height=600
        )
        
        return fig
    
    except Exception as e:
        st.error(f"Error creating feature importance chart: {str(e)}")
        return None

def plot_price_by_feature(df, feature, title=None):
    """Create a box plot of prices grouped by a categorical feature.
    
    Args:
        df: DataFrame with price and feature data
        feature: Name of categorical feature to group by
        title: Title for the chart (optional)
        
    Returns:
        plotly.graph_objects.Figure: Box plot of prices by feature
    """
    try:
        if feature not in df.columns:
            return None
            
        if title is None:
            title = f"Resale Price by {feature.replace('_', ' ').title()}"
            
        fig = px.box(
            df, 
            x=feature, 
            y="resale_price",
            title=title,
            labels={
                feature: feature.replace('_', ' ').title(),
                "resale_price": "Resale Price (SGD)"
            }
        )
        
        fig.update_layout(
            xaxis_title=feature.replace('_', ' ').title(),
            yaxis_title="Resale Price (SGD)",
            height=500
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating box plot: {str(e)}")
        return None

def plot_correlation_heatmap(df, features=None, title="Feature Correlation Heatmap"):
    """Create a correlation heatmap for selected features.
    
    Args:
        df: DataFrame with numerical feature data
        features: List of features to include (optional)
        title: Title for the heatmap (optional)
        
    Returns:
        matplotlib.figure.Figure: Correlation heatmap
    """
    try:
        # Filter to numerical columns if no features specified
        if features is None:
            numeric_df = df.select_dtypes(include=['int64', 'float64'])
        else:
            numeric_df = df[features]
            
        # Compute correlation matrix
        corr_matrix = numeric_df.corr()
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create heatmap
        sns.heatmap(
            corr_matrix, 
            annot=True, 
            cmap='coolwarm', 
            fmt=".2f",
            linewidths=0.5,
            ax=ax
        )
        
        # Set title
        ax.set_title(title, fontsize=16, pad=20)
        
        return fig
    except Exception as e:
        st.error(f"Error creating correlation heatmap: {str(e)}")
        return None