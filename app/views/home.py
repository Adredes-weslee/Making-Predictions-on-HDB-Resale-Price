"""Home page view for the Streamlit application.

This module provides the home page view for the application, which includes
an introduction to the project, overview of the dataset, and highlights of key insights.
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path
import time

def show_home():
    """Display the home page content."""
    # Header
    st.markdown("<h1 class='main-header'>HDB Resale Price Prediction</h1>", unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    ## Welcome to the HDB Resale Price Predictor!
    
    This application helps you explore public housing resale prices in Singapore and make predictions
    based on various property attributes. Using machine learning models trained on historical transaction
    data, you can get estimated prices for HDB flats across different towns, flat types, and more.
    """)
    
    # Main sections with full width
    st.markdown("""
    ### What You Can Do:
    
    **📊 Data Explorer**
    - Visualize resale price trends over time
    - Compare prices across different towns and flat types
    - Explore relationships between property attributes and prices
    
    **🔮 Make Prediction**
    - Input property details to get a predicted resale price
    - See how changing attributes affects the estimated price
    
    **📈 Model Performance**
    - Compare performance of different prediction models
    - Understand which features have the most impact on prices
    - See evaluation metrics for each model
    """)
    
    # Dataset overview section continues as before...
    st.markdown("---")
    st.markdown("## Dataset Overview")
    
    # Load dataset statistics from cache or calculate them
    @st.cache_data
    def get_dataset_stats():
        try:
            # Get data directory
            root_dir = Path(__file__).parent.parent.parent
            data_path = os.path.join(root_dir, 'data', 'processed', 'train_processed_exploratory.csv')
            
            # Check if file exists
            if not os.path.exists(data_path):
                return {
                    "transactions": "N/A",
                    "time_period": "N/A",
                    "towns": "N/A",
                    "price_range": "N/A"
                }
            
            # Load data
            df = pd.read_csv(data_path, low_memory=False)
            
            # Calculate statistics
            stats = {
                "transactions": f"{len(df):,}",
                "time_period": f"{df['year_month'].min()} to {df['year_month'].max()}" if 'year_month' in df.columns else "N/A",
                "towns": f"{df['town'].nunique()}" if 'town' in df.columns else "N/A",
                "price_range": f"${df['resale_price'].min():,.0f} - ${df['resale_price'].max():,.0f}" if 'resale_price' in df.columns else "N/A"
            }
            
            # Store town and flat type options in session state for use in other pages
            if 'town' in df.columns:
                st.session_state['towns'] = sorted(df['town'].unique().tolist())
            
            if 'flat_type' in df.columns:
                st.session_state['flat_types'] = sorted(df['flat_type'].unique().tolist())
                
            return stats
            
        except Exception as e:
            st.error(f"Error loading dataset statistics: {str(e)}")
            return {
                "transactions": "Error",
                "time_period": "Error",
                "towns": "Error",
                "price_range": "Error"
            }
    
    # Get statistics
    stats = get_dataset_stats()
    
    # Display statistics in columns
    cols = st.columns(4)
    with cols[0]:
        st.metric("Total Transactions", stats["transactions"])
    with cols[1]:
        st.metric("Time Period", stats["time_period"])
    with cols[2]:
        st.metric("Number of Towns", stats["towns"])
    with cols[3]:
        st.metric("Price Range", stats["price_range"])
        
    # Project description
    st.markdown("---")
    st.markdown("""
    ## About This Project
    
    This project analyzes HDB resale transactions to build price prediction models. 
    
    The models incorporate various factors that impact property values, including location, property attributes, 
    market conditions, and more. Using machine learning techniques, we've identified the most important factors
    and created models that can estimate resale prices with high accuracy.
    
    ### Data Sources
    
    The dataset used in this project comes from data.gov.sg and includes historical HDB resale transactions.
    
    ### Methodology
    
    We employed several regression models (Linear Regression, Ridge, Lasso) and evaluated their performance.
    The models were trained on historical data and validated using standard metrics like R² and RMSE.
    
    Use the navigation on the left to explore more!
    """)