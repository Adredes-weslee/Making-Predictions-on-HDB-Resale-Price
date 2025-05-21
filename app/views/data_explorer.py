"""Data Explorer view for the Streamlit application.

This module provides interactive visualizations and data exploration capabilities.
Users can explore relationships between features and resale prices through
various charts and filtering options.
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path
from components.visualizations import (
    plot_price_distribution,
    plot_price_trends,
    plot_price_by_feature,
    plot_correlation_heatmap
)

@st.cache_data
def load_exploratory_data():
    """Load the exploratory dataset for visualization.
    
    Returns:
        pandas.DataFrame: Exploratory dataset
    """
    try:
        # Get data directory
        root_dir = Path(__file__).parent.parent.parent
        data_path = os.path.join(root_dir, 'data', 'processed', 'train_processed_exploratory.csv')
        
        # Check if file exists
        if not os.path.exists(data_path):
            st.error(f"Processed data not found at {data_path}")
            return None
        
        # Load data
        df = pd.read_csv(data_path)
        return df
        
    except Exception as e:
        st.error(f"Error loading exploratory data: {str(e)}")
        return None

def show_data_explorer():
    """Display the data explorer page with interactive visualizations."""
    # Header
    st.markdown("<h1 class='main-header'>Data Explorer</h1>", unsafe_allow_html=True)
    
    # Load data
    df = load_exploratory_data()
    
    if df is None:
        st.error("Could not load exploratory data. Please check that the processed data files exist.")
        return
    
    # Filter controls
    with st.expander("Data Filters", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'towns' in st.session_state:
                towns = st.session_state.towns
                selected_towns = st.multiselect(
                    "Towns",
                    options=towns,
                    default=towns[:5] if len(towns) > 5 else towns
                )
            else:
                selected_towns = None
        
        with col2:
            if 'flat_types' in st.session_state:
                flat_types = st.session_state.flat_types
                selected_flat_types = st.multiselect(
                    "Flat Types",
                    options=flat_types,
                    default=flat_types
                )
            else:
                selected_flat_types = None
                
        with col3:
            if 'year_month' in df.columns:
                year_months = sorted(df['year_month'].unique())
                start_date = st.selectbox(
                    "Start Date",
                    options=year_months,
                    index=0
                )
                end_date = st.selectbox(
                    "End Date",
                    options=year_months,
                    index=len(year_months)-1
                )
            else:
                start_date = None
                end_date = None
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_towns:
        filtered_df = filtered_df[filtered_df['town'].isin(selected_towns)]
        
    if selected_flat_types:
        filtered_df = filtered_df[filtered_df['flat_type'].isin(selected_flat_types)]
        
    if start_date and end_date and 'year_month' in filtered_df.columns:
        filtered_df = filtered_df[(filtered_df['year_month'] >= start_date) & 
                                 (filtered_df['year_month'] <= end_date)]
    
    # Show data summary
    st.markdown(f"### Data Summary: {len(filtered_df):,} transactions")
    
    # Tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Price Distribution", "Price Trends", "Price by Feature", "Correlations"])
    
    with tab1:
        # Price distribution
        st.markdown("### Resale Price Distribution")
        price_dist_fig = plot_price_distribution(filtered_df)
        if price_dist_fig:
            st.plotly_chart(price_dist_fig, use_container_width=True)
            
        # Show summary statistics
        st.markdown("#### Price Statistics (SGD)")
        
        stats_cols = st.columns(5)
        stats_cols[0].metric("Mean", f"${filtered_df['resale_price'].mean():,.0f}")
        stats_cols[1].metric("Median", f"${filtered_df['resale_price'].median():,.0f}")
        stats_cols[2].metric("Min", f"${filtered_df['resale_price'].min():,.0f}")
        stats_cols[3].metric("Max", f"${filtered_df['resale_price'].max():,.0f}")
        stats_cols[4].metric("Std Dev", f"${filtered_df['resale_price'].std():,.0f}")
    
    with tab2:
        # Price trends over time
        st.markdown("### Resale Price Trends")
        trends_fig = plot_price_trends(filtered_df)
        if trends_fig:
            st.plotly_chart(trends_fig, use_container_width=True)
        else:
            st.info("Price trend data not available")
            
    with tab3:
        # Price by categorical features
        st.markdown("### Resale Price by Feature")
        
        # Feature selector
        categorical_features = ['town', 'flat_type', 'storey_range', 'flat_model']
        available_features = [f for f in categorical_features if f in filtered_df.columns]
        
        if available_features:
            selected_feature = st.selectbox(
                "Select Feature",
                options=available_features,
                index=0
            )
            
            # Box plot
            box_fig = plot_price_by_feature(filtered_df, selected_feature)
            if box_fig:
                st.plotly_chart(box_fig, use_container_width=True)
        else:
            st.info("No categorical features available")
            
    with tab4:
        # Correlation heatmap
        st.markdown("### Feature Correlations")
        
        # Select numerical features
        numerical_cols = filtered_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if 'resale_price' in numerical_cols:
            numerical_cols.remove('resale_price')
            numerical_cols = ['resale_price'] + numerical_cols
            
        # Allow user to select features for correlation
        if len(numerical_cols) > 2:
            selected_num_features = st.multiselect(
                "Select Features for Correlation Analysis",
                options=numerical_cols,
                default=numerical_cols[:10] if len(numerical_cols) > 10 else numerical_cols
            )
            
            if selected_num_features and len(selected_num_features) >= 2:
                corr_fig = plot_correlation_heatmap(filtered_df, selected_num_features)
                if corr_fig:
                    st.pyplot(corr_fig)
            else:
                st.info("Please select at least 2 features for correlation analysis")
        else:
            st.info("Not enough numerical features for correlation analysis")
    
    # Show raw data sample
    with st.expander("View Raw Data Sample", expanded=False):
        st.dataframe(filtered_df.head(100), use_container_width=True)