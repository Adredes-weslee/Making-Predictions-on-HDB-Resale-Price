"""Prediction view for the Streamlit application.

This module provides an interface for users to input property details and get
price predictions using the pre-trained models.
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path
import joblib
import json
from datetime import datetime

@st.cache_data
def load_towns_and_features():
    """Load unique values for towns and other feature options.
    
    Returns:
        dict: Dictionary with feature options
    """
    try:
        # Get data directory
        root_dir = Path(__file__).parent.parent.parent
        data_path = os.path.join(root_dir, 'data', 'processed', 'train_processed_exploratory.csv')
        
        if not os.path.exists(data_path):
            return {}
            
        # Load a sample of the data
        df = pd.read_csv(data_path, nrows=10000)
        
        # Extract unique values for categorical features
        feature_options = {}
        
        if 'town' in df.columns:
            feature_options['towns'] = sorted(df['town'].unique().tolist())
            
        if 'flat_type' in df.columns:
            feature_options['flat_types'] = sorted(df['flat_type'].unique().tolist())
            
        if 'storey_range' in df.columns:
            feature_options['storey_ranges'] = sorted(df['storey_range'].unique().tolist())
            
        if 'flat_model' in df.columns:
            feature_options['flat_models'] = sorted(df['flat_model'].unique().tolist())
            
        # Get numerical feature ranges
        num_features = ['floor_area_sqm', 'remaining_lease']
        for feat in num_features:
            if feat in df.columns:
                feature_options[f'{feat}_min'] = float(df[feat].min())
                feature_options[f'{feat}_max'] = float(df[feat].max())
                
        return feature_options
        
    except Exception as e:
        st.error(f"Error loading feature options: {str(e)}")
        return {}

def make_prediction(input_data, models_dict, model_type='ridge'):
    """Make a price prediction using the selected model.
    
    Args:
        input_data: Dictionary of property attributes
        models_dict: Dictionary containing loaded models
        model_type: Type of model to use for prediction
        
    Returns:
        float: Predicted price
        dict: Additional information for display
    """
    try:
        # Check if model exists
        if f"{model_type}_model" not in models_dict or models_dict[f"{model_type}_model"] is None:
            return None, {"error": f"Model '{model_type}' not available"}
            
        # Get the model
        model = models_dict[f"{model_type}_model"]
        
        # Prepare input data as DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Make prediction
        predicted_price = model.predict(input_df)[0]
        
        # Get model metrics for context
        model_metrics = models_dict.get(f"{model_type}_metrics", {})
        model_r2 = model_metrics.get('test_r2', 0)
        model_rmse = model_metrics.get('test_rmse', 0)
        
        # Prepare additional information
        info = {
            "predicted_price": predicted_price,
            "model_type": model_type,
            "model_r2": model_r2,
            "model_rmse": model_rmse
        }
        
        return predicted_price, info
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, {"error": str(e)}

def show_prediction():
    """Display the prediction interface."""
    # Header
    st.markdown("<h1 class='main-header'>Make Prediction</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    Enter property details below to get a predicted resale price. The prediction is based on
    historical transaction data and uses machine learning models to estimate the most likely price.
    """)
    
    # Load feature options
    feature_options = load_towns_and_features()
    
    # Ensure models are loaded
    if 'models' not in st.session_state:
        st.error("Models not loaded. Please refresh the application.")
        return
        
    models_dict = st.session_state['models']
    
    # Form with property inputs
    with st.form("prediction_form"):
        # Layout in columns
        col1, col2 = st.columns(2)
        
        with col1:
            # Location details
            st.subheader("Location")
            
            town = st.selectbox(
                "Town",
                options=feature_options.get('towns', ['ANG MO KIO', 'BEDOK', 'BISHAN']),
                index=0
            )
            
            # Property details
            st.subheader("Property Details")
            
            flat_type = st.selectbox(
                "Flat Type",
                options=feature_options.get('flat_types', ['1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE']),
                index=3
            )
            
            flat_model = st.selectbox(
                "Flat Model",
                options=feature_options.get('flat_models', ['Improved', 'New Generation', 'Standard']),
                index=0
            )
            
            storey_range = st.selectbox(
                "Storey Range",
                options=feature_options.get('storey_ranges', ['01 TO 03', '04 TO 06', '07 TO 09', '10 TO 12', '13 TO 15', '16 TO 18', '19 TO 21']),
                index=2
            )
            
        with col2:
            # Physical attributes
            st.subheader("Physical Attributes")
            
            floor_area_min = feature_options.get('floor_area_sqm_min', 30)
            floor_area_max = feature_options.get('floor_area_sqm_max', 200)
            
            floor_area_sqm = st.slider(
                "Floor Area (sqm)",
                min_value=floor_area_min,
                max_value=floor_area_max,
                value=90.0,
                step=1.0
            )
            
            # Lease details
            st.subheader("Lease Information")
            
            lease_min = feature_options.get('remaining_lease_min', 40)
            lease_max = feature_options.get('remaining_lease_max', 99)
            
            remaining_lease = st.slider(
                "Remaining Lease (years)",
                min_value=lease_min,
                max_value=lease_max,
                value=70,
                step=1
            )
            
            # Transaction date
            st.subheader("Transaction Details")
            
            transaction_date = st.date_input(
                "Transaction Date",
                value=datetime.now().date()
            )
            
            # Model selection
            model_type = st.selectbox(
                "Prediction Model",
                options=["ridge", "linear", "lasso"],
                index=0,
                help="Select which model to use for prediction"
            )
        
        # Submit button
        submit = st.form_submit_button("Predict Price")
    
    # Make prediction when form is submitted
    if submit:
        # Prepare input data - this would need to match the format expected by your model
        # This is a simplified example - your actual preprocessing might be more complex
        input_data = {
            'town': town,
            'flat_type': flat_type, 
            'flat_model': flat_model,
            'storey_range': storey_range,
            'floor_area_sqm': floor_area_sqm,
            'remaining_lease': remaining_lease,
            'transaction_month': transaction_date.month,
            'transaction_year': transaction_date.year
        }
        
        # Get the prediction
        predicted_price, info = make_prediction(input_data, models_dict, model_type)
        
        # Display prediction
        if predicted_price is not None:
            st.markdown("---")
            st.markdown("## Prediction Results")
            
            # Show the prediction
            st.markdown(f"### Estimated Resale Price")
            st.markdown(f"<h1 style='color:#1E88E5;'>${predicted_price:,.2f}</h1>", unsafe_allow_html=True)
            
            # Show model information
            st.markdown("### Model Information")
            st.markdown(f"* **Model Type:** {info['model_type'].capitalize()}")
            st.markdown(f"* **Model Accuracy (R²):** {info['model_r2']:.4f}")
            st.markdown(f"* **Error Margin (RMSE):** ${info['model_rmse']:,.2f}")
            
            # Show input summary
            st.markdown("### Property Summary")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Town:** {town}")
                st.markdown(f"**Flat Type:** {flat_type}")
                st.markdown(f"**Flat Model:** {flat_model}")
                st.markdown(f"**Storey Range:** {storey_range}")
            
            with col2:
                st.markdown(f"**Floor Area:** {floor_area_sqm} sqm")
                st.markdown(f"**Remaining Lease:** {remaining_lease} years")
                st.markdown(f"**Transaction Date:** {transaction_date.strftime('%B %Y')}")
        else:
            st.error(f"Could not make prediction: {info.get('error', 'Unknown error')}")