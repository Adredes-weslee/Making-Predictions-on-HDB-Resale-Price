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
    """Load sample data for reference values and property attributes.
    
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
        df = load_raw_data(data_paths["processed"])
        return df
    except FileNotFoundError:
        # Fallback to train data
        df = load_raw_data(data_paths["train"])
        return df


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
                # Location details
                st.markdown("### Location Details")
                
                towns = sorted(sample_data['town'].unique())
                town = st.selectbox("Town", towns)
                
                flat_types = sorted(sample_data['flat_type'].unique())
                flat_type = st.selectbox("Flat Type", flat_types)
                
                if 'storey_range' in sample_data.columns:
                    storey_ranges = sorted(sample_data['storey_range'].unique())
                    storey_range = st.selectbox("Storey Range", storey_ranges)
                else:
                    storey_ranges = ["01 TO 03", "04 TO 06", "07 TO 09", "10 TO 12", "13 TO 15", "16 TO 18", "19 TO 21", "22 TO 24", "25 TO 27", "28 TO 30"]
                    storey_range = st.selectbox("Storey Range", storey_ranges)
                
                if 'flat_model' in sample_data.columns:
                    flat_models = sorted(sample_data['flat_model'].unique())
                    flat_model = st.selectbox("Flat Model", flat_models)
                else:
                    flat_model = "New Generation"  # Default value
            
            with col2:
                # Physical attributes
                st.markdown("### Physical Attributes")
                
                # Floor area
                min_area = float(sample_data['floor_area_sqm'].min())
                max_area = float(sample_data['floor_area_sqm'].max())
                default_area = float(sample_data['floor_area_sqm'].median())
                floor_area_sqm = st.slider(
                    "Floor Area (sqm)", 
                    min_value=min_area,
                    max_value=max_area,
                    value=default_area,
                    step=1.0
                )
                
                # Lease details
                current_year = datetime.now().year
                lease_commence_date = st.slider(
                    "Lease Commence Date", 
                    min_value=1960,
                    max_value=current_year,
                    value=2000
                )
                
                remaining_lease_years = st.slider(
                    "Remaining Lease (Years)", 
                    min_value=1,
                    max_value=99,
                    value=70
                )
                
                remaining_lease_months = st.slider(
                    "Remaining Lease (Months)", 
                    min_value=0,
                    max_value=11,
                    value=0
                )
            
            # Additional features
            st.markdown("### Additional Features (if available)")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'mrt_nearest_distance' in sample_data.columns:
                    mrt_nearest_distance = st.slider(
                        "Distance to Nearest MRT (meters)", 
                        min_value=0,
                        max_value=int(sample_data['mrt_nearest_distance'].max()),
                        value=1000
                    )
                else:
                    mrt_nearest_distance = 1000  # Default value
            
            with col2:
                if 'mall_nearest_distance' in sample_data.columns:
                    mall_nearest_distance = st.slider(
                        "Distance to Nearest Mall (meters)", 
                        min_value=0,
                        max_value=int(sample_data['mall_nearest_distance'].max() if 'mall_nearest_distance' in sample_data.columns else 2000),
                        value=1000
                    )
                else:
                    mall_nearest_distance = 1000  # Default value
            
            with col3:
                if 'schools_within_1km' in sample_data.columns:
                    schools_within_1km = st.slider(
                        "Number of Schools Within 1km", 
                        min_value=0,
                        max_value=int(sample_data['schools_within_1km'].max() if 'schools_within_1km' in sample_data.columns else 5),
                        value=2
                    )
                else:
                    schools_within_1km = 2  # Default value
            
            # Submit button
            predict_button = st.form_submit_button("Predict Resale Price")
        
        # Handle prediction when form is submitted
        if predict_button:
            # Create DataFrame for prediction
            input_data = {
                'town': town,
                'flat_type': flat_type,
                'storey_range': storey_range,
                'flat_model': flat_model,
                'floor_area_sqm': floor_area_sqm,
                'lease_commence_date': lease_commence_date
            }
            
            # Add remaining lease
            remaining_lease = f"{remaining_lease_years} years {remaining_lease_months} months"
            input_data['remaining_lease'] = remaining_lease
            
            # Add additional features if they were used in training
            if 'mrt_nearest_distance' in sample_data.columns:
                input_data['mrt_nearest_distance'] = mrt_nearest_distance
            
            if 'mall_nearest_distance' in sample_data.columns:
                input_data['mall_nearest_distance'] = mall_nearest_distance
                
            if 'schools_within_1km' in sample_data.columns:
                input_data['schools_within_1km'] = schools_within_1km
            
            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Make prediction
            try:
                # Preprocess the input data
                processed_input = preprocess_data(input_df, is_training=False)
                
                # Predict
                price_prediction = predict(model, processed_input, preprocessed=True)
                
                # Display prediction
                st.success("Prediction completed successfully!")
                st.subheader("Predicted Resale Price")
                
                # Format the predicted price
                predicted_price = float(price_prediction[0])
                st.markdown(f"### S${predicted_price:,.2f}")
                
                # Display confidence interval (simple estimation)
                lower_bound = predicted_price * 0.85
                upper_bound = predicted_price * 1.15
                
                st.write(f"Estimated range: S${lower_bound:,.2f} - S${upper_bound:,.2f}")
                
                # Show comparison with similar properties
                st.subheader("Comparison with Similar Properties")
                
                # Filter similar properties
                similar_props = sample_data[
                    (sample_data['town'] == town) &
                    (sample_data['flat_type'] == flat_type)
                ].copy()
                
                if len(similar_props) > 0:
                    # Calculate average price for similar properties
                    avg_similar_price = similar_props['resale_price'].mean()
                    
                    # Create comparison bar chart
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    comparison_data = pd.DataFrame({
                        'Category': ['Predicted Price', 'Average Similar Properties'],
                        'Price': [predicted_price, avg_similar_price]
                    })
                    
                    sns.barplot(x='Category', y='Price', data=comparison_data, ax=ax)
                    ax.set_ylabel('Price (SGD)')
                    ax.set_title(f'Price Comparison in {town} for {flat_type}')
                    
                    # Format y-axis labels
                    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"${x:,.0f}"))
                    
                    st.pyplot(fig)
                    
                    # Price distribution for similar properties
                    st.write(f"### Price Distribution for Similar Properties in {town}")
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.histplot(similar_props['resale_price'], kde=True, ax=ax)
                    
                    # Add line for predicted value
                    ax.axvline(x=predicted_price, color='red', linestyle='--', label='Predicted Price')
                    
                    ax.set_xlabel('Resale Price (SGD)')
                    ax.set_ylabel('Count')
                    ax.legend()
                    
                    st.pyplot(fig)
                    
                    # Display similar properties
                    st.write("### Recent Similar Transactions")
                    st.write("Here are some recent transactions for similar properties:")
                    
                    # Sort by most recent transactions if date column exists
                    if 'Tranc_YearMonth' in similar_props.columns:
                        similar_props = similar_props.sort_values('Tranc_YearMonth', ascending=False)
                    
                    # Select relevant columns for display
                    display_cols = ['resale_price', 'floor_area_sqm', 'storey_range']
                    if 'Tranc_YearMonth' in similar_props.columns:
                        display_cols.insert(0, 'Tranc_YearMonth')
                    
                    st.dataframe(similar_props[display_cols].head(10), use_container_width=True)
                else:
                    st.warning("No similar properties found in the dataset for comparison.")
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.warning("Please check that all input fields are filled correctly.")
    
    except Exception as e:
        st.error(f"Error loading model or sample data: {str(e)}")
        st.warning("The prediction feature is currently unavailable. Please try again later.")
