"""Home page component for the Streamlit application.

This module defines the homepage UI for the HDB Resale Price Prediction Streamlit
application. It displays an informative landing page with project overview,
key statistics about the dataset, usage instructions, and information about
the prediction model.

The homepage serves as the main entry point for users, providing navigation
guidance and context about the HDB resale market in Singapore.

Typical usage:
    >>> import streamlit as st
    >>> from app.pages.home import show_home
    >>> show_home()
"""
import streamlit as st
import pandas as pd
import os
from pathlib import Path

# Import utility functions
from src.data.loader import get_data_paths, load_raw_data


def show_home():
    """Display the home page of the application.
    
    This function renders the complete homepage UI with multiple sections:
    1. Introduction and welcome message
    2. Quick statistics about the dataset (if available)
    3. Application usage instructions
    4. Background information about the prediction model
    5. Project footer with attribution
    
    The function attempts to load and display summary statistics from the
    processed dataset. If data loading fails, appropriate error messages
    are shown without disrupting the rest of the UI.
    
    Returns:
        None: The function renders Streamlit UI elements directly.
        
    Raises:
        No exceptions are raised as errors are handled internally with
        appropriate UI feedback.
    
    Example:
        >>> show_home()
        # Renders the complete homepage in the Streamlit app
    """
    st.title("HDB Resale Price Prediction")
    
    # Introduction section
    st.markdown("""
    ## Welcome to the HDB Resale Price Analytics Dashboard!
    
    This application provides insights and predictions for Housing Development Board (HDB) 
    resale flat prices in Singapore. The app is built on a machine learning model trained on 
    historical HDB resale transactions.
    
    ### What you can do with this application:
    
    - **Explore Data**: Analyze historical HDB resale transactions with interactive visualizations
    - **Predict Prices**: Get estimated resale price for a flat based on its attributes
    - **Model Insights**: Understand which factors influence HDB resale prices the most
    
    ### About HDB Resale Flats
    
    The Housing & Development Board (HDB) is Singapore's public housing authority. Over 80% of 
    Singapore's population lives in HDB flats, making them a crucial component of the housing market.
    The resale prices of these flats are influenced by various factors including location, flat type,
    floor area, and remaining lease.
    """)
    
    # Quick stats
    try:
        # Load data summary
        data_paths = get_data_paths()
        processed_dir = data_paths["processed"]
        # Use the train_processed.csv file specifically
        processed_file = os.path.join(processed_dir, "train_processed.csv")
        df = load_raw_data(processed_file)
        
        # Display quick stats in columns
        st.subheader("Quick Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        
        with col2:
            avg_price = df["resale_price"].mean()
            st.metric("Average Price", f"${avg_price:,.2f}")
        
        with col3:
            num_towns = df["town"].nunique()
            st.metric("Towns", num_towns)
        
        with col4:
            recent_year = df["year"].max() if "year" in df.columns else df["Tranc_YearMonth"].dt.year.max() if "Tranc_YearMonth" in df.columns else "N/A"
            st.metric("Latest Data Year", recent_year)
    
    except Exception as e:
        st.warning("Could not load data statistics. Please check the data files.")
        st.error(f"Error: {e}")
    
    # How to use section
    st.markdown("""
    ## How to Use This Application
    
    1. Use the navigation menu on the sidebar to switch between different sections
    2. In the **Data Explorer**, you can analyze HDB resale data through various visualizations
    3. In the **Make Prediction** section, enter details about an HDB flat to get a price estimate
    4. The **Model Insights** section provides information about the features that influence prices
    
    ### Background on the Model
    
    The prediction model is built using machine learning techniques with features including:
    
    - Location-based features (town, planning area)
    - Property attributes (flat type, floor area)
    - Age and lease-related features
    - Nearby amenities and facilities
    - Historical transaction trends
    
    The model is regularly updated to reflect the latest market trends and provides reliable predictions 
    for HDB resale flat prices in Singapore.
    """)
    
    # Footer
    st.markdown("---")
    st.markdown("### About this project")
    st.markdown(
        "This application was developed as part of a data science project to analyze and "
        "predict HDB resale prices in Singapore. The source code is available on GitHub."
    )
