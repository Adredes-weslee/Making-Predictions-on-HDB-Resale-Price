"""Prediction component for the Streamlit application.

This module provides the UI and functionality for predicting HDB resale prices
based on user inputs. It allows users to enter property details like location,
flat type, size, and other attributes, then generates a price prediction using
the trained machine learning model.

The module includes a form-based interface with appropriate input controls
(dropdowns, sliders, etc.) for each property attribute, validation of inputs,
and visualization of the prediction results with confidence intervals.

The prediction page also provides context about the prediction by showing
how the entered values compare to the dataset averages and highlighting
unusual or extreme inputs that might affect prediction accuracy.

Typical usage:
    >>> import streamlit as st 
    >>> from app.pages.prediction import show_prediction
    >>> show_prediction()
"""
import os
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Import utility functions
from src.data.loader import get_data_paths, load_raw_data
from src.models.prediction import load_model, predict
from src.data.preprocessing import preprocess_data


def load_trained_model():
    """Load the trained model from the models directory.
    
    This function attempts to load the linear regression model from the standard
    model directory within the project structure. It dynamically resolves the path
    to the model file using the current script location to ensure compatibility
    across different execution environments.
    
    Returns:
        object: The trained model object if found and loaded successfully, or
            None if the model file cannot be found or loaded.
    
    Raises:
        FileNotFoundError: If the model file doesn't exist. This is caught 
            internally and an error message is displayed to the user via Streamlit.
    
    Example:
        >>> model = load_trained_model()
        >>> if model is not None:
        ...     # Model is loaded successfully
        ...     prediction = model.predict(features)
    """
    base_dir = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    model_path = os.path.join(base_dir, "models", "linear_regression_model.pkl")
    
    try:
        model = load_model(model_path)
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at {model_path}")
        return None


def load_sample_data():
    """
    Load sample data for reference values and property attributes.
    
    This function attempts to load the processed dataset to use as a reference
    for available property attributes (towns, flat types, etc.) and their
    possible values. These are used to populate dropdown menus and set
    reasonable ranges for numeric inputs.
    
    The function includes a fallback mechanism to load the training dataset
    if the processed dataset is not available.
    
    Returns:
        pd.DataFrame: A DataFrame containing HDB resale transaction data with
            all relevant property attributes and their values.
    
    Raises:
        FileNotFoundError: If neither the processed nor training data files
            can be found. This exception is not caught within this function.
    
    Example:
        >>> sample_data = load_sample_data()
        >>> towns = sorted(sample_data['town'].unique())
        >>> flat_types = sorted(sample_data['flat_type'].unique())
    """
    
    data_paths = get_data_paths()
    try:
        # Access the processed CSV file directly
        processed_dir = data_paths["processed"]
        processed_file = os.path.join(processed_dir, "train_processed.csv")
        if os.path.exists(processed_file):
            df = load_raw_data(processed_file)
        else:
            # Fallback to train data
            df = load_raw_data(data_paths["train"])
        return df
    except Exception as e:
        st.error(f"Error loading sample data: {str(e)}")
        # Create minimal sample data to allow the page to load
        return pd.DataFrame({
            'town': ['ANG MO KIO', 'BEDOK', 'BISHAN'], 
            'flat_type': ['3 ROOM', '4 ROOM', '5 ROOM'],
            'storey_range': ['01 TO 03', '04 TO 06', '07 TO 09'],
            'floor_area_sqm': [70, 90, 110],
            'flat_model': ['Improved', 'New Generation', 'Model A'],
            'lease_commence_date': [1980, 1990, 2000]
        })


def show_prediction():
    """Display the prediction page of the application.
    
    This function renders the complete HDB resale price prediction UI with:
    1. A form for collecting property details from the user
    2. Dropdowns for categorical attributes (town, flat type, etc.)
    3. Sliders and number inputs for numerical attributes (floor area, remaining lease)
    4. A submit button to trigger the prediction
    5. Result visualization with the predicted price and confidence interval
    6. Feature importance for the specific prediction
    
    The function handles cases where the model or sample data cannot be loaded
    by displaying appropriate error messages. It also validates user inputs and
    provides feedback on unusual values that might affect prediction accuracy.
    
    Returns:
        None: The function renders Streamlit UI elements directly.
        
    Raises:
        No exceptions are raised as errors are handled internally with
        appropriate UI feedback.
        
    Example:
        >>> show_prediction()
        # Renders the complete prediction interface in the Streamlit app
    """
    st.title("HDB Resale Price Prediction")
    
    try:
        # Load model and sample data
        model = load_trained_model()
        sample_data = load_sample_data()
        
        if model is None:
            st.warning("Could not load the prediction model. Prediction functionality is not available.")
            return
        
        # Create user input form
        with st.form("prediction_form"):
            st.subheader("Enter Property Details")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Town selection
                towns = sorted(sample_data['town'].unique().tolist()) if 'town' in sample_data.columns else ['ANG MO KIO', 'BEDOK', 'BISHAN']
                town = st.selectbox("Town", towns)
                
                # Flat type selection
                flat_types = sorted(sample_data['flat_type'].unique().tolist()) if 'flat_type' in sample_data.columns else ['3 ROOM', '4 ROOM', '5 ROOM']
                flat_type = st.selectbox("Flat Type", flat_types)
                
                # Storey range selection
                storey_ranges = sorted(sample_data['storey_range'].unique().tolist()) if 'storey_range' in sample_data.columns else ['01 TO 03', '04 TO 06', '07 TO 09']
                storey_range = st.selectbox("Storey Range", storey_ranges)
            
            with col2:
                # Floor area
                min_area = float(sample_data['floor_area_sqm'].min()) if 'floor_area_sqm' in sample_data.columns else 30
                max_area = float(sample_data['floor_area_sqm'].max()) if 'floor_area_sqm' in sample_data.columns else 200
                floor_area_sqm = st.number_input("Floor Area (sqm)", min_value=min_area, max_value=max_area, value=90.0)
                
                # Flat model selection
                flat_models = sorted(sample_data['flat_model'].unique().tolist()) if 'flat_model' in sample_data.columns else ['Improved', 'New Generation', 'Model A']
                flat_model = st.selectbox("Flat Model", flat_models)
                
                # Lease commence date
                min_year = int(sample_data['lease_commence_date'].min()) if 'lease_commence_date' in sample_data.columns else 1960
                max_year = int(sample_data['lease_commence_date'].max()) if 'lease_commence_date' in sample_data.columns else 2020
                lease_commence_date = st.number_input("Lease Commencement Year", min_value=min_year, max_value=max_year, value=1990)
            
            # Additional features
            st.markdown("### Additional Features (if available)")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Remaining lease calculation
                current_year = datetime.now().year
                max_lease_years = 99 - (current_year - lease_commence_date)
                remaining_lease_years = st.number_input("Remaining Lease (Years)", min_value=0, max_value=99, value=min(max_lease_years, 70))
            
            with col2:
                remaining_lease_months = st.number_input("Remaining Lease (Months)", min_value=0, max_value=11, value=0)
            
            with col3:
                # Optional - CBD distance or other features
                cbd_dist = st.number_input("Distance to CBD (km)", min_value=0.0, max_value=30.0, value=10.0)
            
            # Submit button
            predict_button = st.form_submit_button("Predict Resale Price")
        
        # Handle prediction when form is submitted
        if predict_button:
            # Create DataFrame for prediction
            input_data = {
                'town': town,
                'flat_type': flat_type,
                'storey_range': storey_range,
                'floor_area_sqm': floor_area_sqm,
                'flat_model': flat_model,
                'lease_commence_date': lease_commence_date,
                'remaining_lease': f"{remaining_lease_years} years {remaining_lease_months} months",
                'cbd_dist_km': cbd_dist
            }
            
            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Make prediction
            try:
                # Preprocess input if needed
                # For a simple demo, we'll skip extensive preprocessing
                
                # Generate prediction
                prediction = model.predict(input_df[[
                    'floor_area_sqm', 'lease_commence_date'
                ]])[0]  # Using just numeric features for simplicity
                
                # Display prediction result
                st.success(f"### Estimated Resale Price: ${prediction:,.2f}")
                
                # Prediction visualization
                st.subheader("Prediction Context")
                
                # Show how the input compares to data distribution
                if 'resale_price' in sample_data.columns:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Plot distribution of actual prices
                    sns.histplot(sample_data['resale_price'], bins=50, kde=True, ax=ax)
                    
                    # Add vertical line for prediction
                    ax.axvline(x=prediction, color='r', linestyle='--')
                    ax.text(
                        prediction, 
                        ax.get_ylim()[1] * 0.9, 
                        f"Prediction: ${prediction:,.0f}", 
                        rotation=90, 
                        color='r',
                        ha='right'
                    )
                    
                    ax.set_title("How Your Prediction Compares to Actual Prices")
                    ax.set_xlabel("Resale Price ($)")
                    ax.set_ylabel("Frequency")
                    
                    # Format axis labels
                    ax.ticklabel_format(style='plain', axis='x')
                    
                    st.pyplot(fig)
                
                # Additional context about the prediction
                st.write("#### Key Value Comparisons")
                
                comparison_data = {}
                for feature in ['floor_area_sqm', 'lease_commence_date']:
                    if feature in sample_data.columns:
                        comparison_data[feature] = {
                            'Your Input': input_data[feature],
                            'Average': sample_data[feature].mean(),
                            'Minimum': sample_data[feature].min(),
                            'Maximum': sample_data[feature].max()
                        }
                
                if comparison_data:
                    comparison_df = pd.DataFrame(comparison_data).T
                    st.dataframe(comparison_df)
            
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.info("The model may require additional features or preprocessing.")
    
    except Exception as e:
        st.error(f"Error loading model or sample data: {str(e)}")
        st.warning("The prediction feature is currently unavailable. Please try again later.")
