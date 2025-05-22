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
import pickle
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
        data_path = os.path.join(root_dir, 'data', 'processed', 'train_pipeline_processed.csv')
        
        if not os.path.exists(data_path):
            # Fall back to exploratory file if pipeline file doesn't exist
            data_path = os.path.join(root_dir, 'data', 'processed', 'train_processed_exploratory.csv')
            
        # Load a sample of the data with low_memory=False
        df = pd.read_csv(data_path, nrows=10000, low_memory=False)
        
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
            
        # Get numerical feature ranges - with additional error handling
        num_features = ['floor_area_sqft', 'remaining_lease']
        for feat in num_features:
            if feat in df.columns:
                try:
                    # Explicitly convert to float to ensure single values
                    feature_options[f'{feat}_min'] = float(df[feat].min())
                    feature_options[f'{feat}_max'] = float(df[feat].max())
                except Exception as e:
                    # Use safe defaults if conversion fails
                    if feat == 'floor_area_sqft':
                        feature_options[f'{feat}_min'] = 300.0
                        feature_options[f'{feat}_max'] = 2000.0
                    elif feat == 'remaining_lease':
                        feature_options[f'{feat}_min'] = 40.0
                        feature_options[f'{feat}_max'] = 99.0
                
        return feature_options
        
    except Exception as e:
        st.error(f"Error loading feature options: {str(e)}")
        return {}

def prepare_features_for_prediction(input_data, model_features):
    """
    Prepare input data for prediction by ensuring all required columns exist.
    
    Args:
        input_data: Dictionary or DataFrame of input features
        model_features: Feature info from the loaded model
        
    Returns:
        pd.DataFrame: DataFrame with all required columns
    """
    # Convert to DataFrame if it's a dictionary
    if isinstance(input_data, dict):
        input_df = pd.DataFrame([input_data])
    else:
        input_df = input_data.copy()
    
    # Get the list of required columns from model_features if available
    required_columns = model_features.get('all_features', None) if model_features else None
    
    # If we don't have the feature list, use the default list (as you already have)
    if not required_columns:
        # Your existing default columns list is fine here
        required_columns = [
            'town', 'flat_type', 'flat_model', 'storey_range', 'floor_area_sqft', 'floor_area_sqm', 'remaining_lease',
            'transaction_month', 'transaction_year', 'Hawker_Nearest_Distance', 'bus_stop_longitude',
            'sec_sch_nearest_dist', 'block', 'hawker_market_stalls', 'lease_commence_date', '1room_sold',
            'lower', 'Tranc_YearMonth', 'Latitude', 'Hawker_Within_1km', 'exec_sold', 'upper', 'postal',
            'hdb_age', 'commercial', '5room_sold', 'affiliation', 'cutoff_point', 'bus_interchange',
            'mid', 'pri_sch_nearest_distance', 'Mall_Within_500m', 'bus_stop_latitude', 'multistorey_carpark',
            'mrt_longitude', 'residential', 'Longitude', 'Tranc_Month', 'Tranc_Year',
            'address', 'sec_sch_longitude', 'vacancy', 'market_hawker', '3room_sold', '3room_rental',
            'Hawker_Within_2km', 'mrt_interchange', 'bus_stop_name', '2room_sold',
            'bus_stop_nearest_distance', 'mrt_nearest_distance', 'hawker_food_stalls',
            'studio_apartment_sold', 'pri_sch_latitude', 'sec_sch_latitude', 'pri_sch_name',
            'mid_storey', 'precinct_pavilion', 'street_name', 'other_room_rental', '4room_sold',
            'Mall_Within_1km', '2room_rental', 'id', 'mrt_name', 'planning_area', 'mrt_latitude',
            'pri_sch_affiliation', 'total_dwelling_units', 'Mall_Nearest_Distance', 'max_floor_lvl',
            'pri_sch_longitude', 'Mall_Within_2km', 'year_completed', 'multigen_sold', 'Hawker_Within_500m',
            'full_flat_type', '1room_rental', 'sec_sch_name'
        ]
    
    # Get categorical features if available
    categorical_features = model_features.get('categorical_features', []) if model_features else []
    
    # Add missing columns with numeric default values (0 for everything)
    for col in required_columns:
        if col not in input_df.columns:
            # Use 0 for all missing columns
            # Pipeline will handle transformations for categorical columns
            input_df[col] = 0
    
    # Handle categorical values that exist in input data
    # Explicitly convert categorical columns to string type since we provided numeric defaults
    for col in categorical_features:
        if col in input_df.columns:
            input_df[col] = input_df[col].astype(str)
    
    return input_df

def make_prediction(input_data, models_dict, model_type='ridge'):
    """Make a price prediction using the selected model."""
    try:
        # Check if model exists
        pipeline_key = f"{model_type}_pipeline"
        model_key = f"{model_type}_model"
        features_key = f"{model_type}_features"
        
        # Get model features if available
        model_features = models_dict.get(features_key, None)
        
        # First try to use pipeline if available (preferred method)
        if pipeline_key in models_dict and models_dict[pipeline_key] is not None:
            # Get the pipeline
            pipeline = models_dict[pipeline_key]
            
            # Prepare input data as DataFrame with all required columns
            input_df = prepare_features_for_prediction(input_data, model_features)
            
            # Use pipeline to transform and predict
            predicted_price = pipeline.predict(input_df)[0]
        
        # Fall back to direct model prediction if pipeline not available
        elif model_key in models_dict and models_dict[model_key] is not None:
            # Get the model
            model = models_dict[model_key]
            
            # Prepare input data
            input_df = prepare_features_for_prediction(input_data, model_features)
            
            # Make prediction
            predicted_price = model.predict(input_df)[0]
        
        else:
            return None, {"error": f"Model '{model_type}' not available"}
        
        # Get model metrics for context - with extra safety check
        model_metrics = models_dict.get(f"{model_type}_metrics", {})
        if model_metrics is None:  # Extra defensive check
            model_metrics = {}
            
        # Now safely access metrics with defaults
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
        import traceback
        st.error(traceback.format_exc())
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
            # Physical attributes section in col2:
            st.subheader("Physical Attributes")

            # Fix for the type mismatch error
            try:
                # Get floor area min/max, ensuring they are single float values
                floor_area_min = float(feature_options.get('floor_area_sqft_min', 300))
                floor_area_max = float(feature_options.get('floor_area_sqft_max', 2000))
                
                # Ensure min doesn't exceed max
                if floor_area_min > floor_area_max:
                    floor_area_min, floor_area_max = 300, 2000
                
                # Default value must be a single number, not a list
                default_area = 970  # Default value (~90 sqm converted to sqft)
                
                # Ensure default is within range
                default_area = max(floor_area_min, min(default_area, floor_area_max))
                
                floor_area_sqft = st.slider(
                    "Floor Area (sqft)",
                    min_value=floor_area_min,
                    max_value=floor_area_max,
                    value=default_area,
                    step=10.0
                )
            except Exception as e:
                # Fallback to safe defaults if any error occurs
                st.warning(f"Using default area range due to: {str(e)}")
                floor_area_sqft = st.slider(
                    "Floor Area (sqft)",
                    min_value=300.0,
                    max_value=2000.0,
                    value=970.0,
                    step=10.0
                )
                        
            # Lease details
            st.subheader("Lease Information")
            
            lease_min = feature_options.get('remaining_lease_min', 40)
            lease_max = feature_options.get('remaining_lease_max', 99)
            
            remaining_lease = st.slider(
                "Remaining Lease (years)",
                min_value=lease_min,
                max_value=lease_max,
                value=70.0,
                step=1.0
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
        # Prepare input data - using floor_area_sqft instead of floor_area_sqm
        input_data = {
            'town': town,
            'flat_type': flat_type, 
            'flat_model': flat_model,
            'storey_range': storey_range,
            'floor_area_sqft': floor_area_sqft,  # Changed from floor_area_sqm
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
                st.markdown(f"**Floor Area:** {floor_area_sqft} sqft")  # Changed from sqm to sqft
                st.markdown(f"**Remaining Lease:** {remaining_lease} years")
                st.markdown(f"**Transaction Date:** {transaction_date.strftime('%B %Y')}")
        else:
            st.error(f"Could not make prediction: {info.get('error', 'Unknown error')}")