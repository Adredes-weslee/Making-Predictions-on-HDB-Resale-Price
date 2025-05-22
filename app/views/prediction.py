"""Prediction view for the Streamlit application.

This module provides an interface for users to input property details and get
price predictions using the pre-trained models.
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import joblib
import pickle
import json
from datetime import datetime

# Add the src directory to Python path to import preprocessing functions
root_dir = Path(__file__).parent.parent.parent
src_path = os.path.join(root_dir, 'src')
if src_path not in sys.path:
    sys.path.append(src_path)

from src.data.preprocessing_pipeline import create_date_features, create_age_features

@st.cache_data
def load_towns_and_features():
    """Load unique values for towns and other feature options."""
    try:
        # Get data directory
        root_dir = Path(__file__).parent.parent.parent
        data_path = os.path.join(root_dir, 'data', 'processed', 'train_pipeline_processed.csv')
        
        if not os.path.exists(data_path):
            # Fall back to exploratory file if pipeline file doesn't exist
            data_path = os.path.join(root_dir, 'data', 'processed', 'train_processed_exploratory.csv')
            
        # Load a sample of the data
        df = pd.read_csv(data_path, nrows=10000, low_memory=False)
        
        # Extract unique values for categorical features
        feature_options = {}
        
        # Categorical features from your reduced dataset
        categorical_cols = ['town', 'flat_type', 'flat_model', 'storey_range', 'mrt_name']
        
        for col in categorical_cols:
            if col in df.columns:
                feature_options[col] = sorted(df[col].unique().tolist())
        
        # Numerical feature ranges
        numerical_cols = ['floor_area_sqm', 'lease_commence_date', 'hdb_age', 'max_floor_lvl', 
                         'Mall_Nearest_Distance', 'Hawker_Nearest_Distance', 'mrt_nearest_distance', 
                         'bus_stop_nearest_distance', 'pri_sch_nearest_distance', 'sec_sch_nearest_dist', 
                         'cutoff_point']
        
        for col in numerical_cols:
            if col in df.columns:
                try:
                    feature_options[f'{col}_min'] = float(df[col].min())
                    feature_options[f'{col}_max'] = float(df[col].max())
                    feature_options[f'{col}_mean'] = float(df[col].mean())
                except:
                    # Set safe defaults
                    defaults = {
                        'floor_area_sqm': (28, 186, 90),
                        'lease_commence_date': (1960, 2020, 1995),
                        'hdb_age': (5, 65, 29),
                        'max_floor_lvl': (3, 50, 12),
                        'Mall_Nearest_Distance': (0, 5000, 800),
                        'Hawker_Nearest_Distance': (0, 3000, 300),
                        'mrt_nearest_distance': (0, 5000, 600),
                        'bus_stop_nearest_distance': (0, 1000, 150),
                        'pri_sch_nearest_distance': (0, 3000, 400),
                        'sec_sch_nearest_dist': (0, 5000, 500),
                        'cutoff_point': (150, 280, 200)
                    }
                    if col in defaults:
                        feature_options[f'{col}_min'] = defaults[col][0]
                        feature_options[f'{col}_max'] = defaults[col][1]
                        feature_options[f'{col}_mean'] = defaults[col][2]
        
        return feature_options
        
    except Exception as e:
        st.error(f"Error loading feature options: {str(e)}")
        return {}

def prepare_features_for_prediction(input_data):
    """Prepare input data using the SAME preprocessing pipeline as training."""
    if isinstance(input_data, dict):
        input_df = pd.DataFrame([input_data])
    else:
        input_df = input_data.copy()
    
    # Create Tranc_YearMonth column for preprocessing pipeline
    if 'year' in input_df.columns and 'month' in input_df.columns:
        input_df['Tranc_YearMonth'] = pd.to_datetime(
            input_df[['year', 'month']].assign(day=1)
        )
    
    # Apply the SAME preprocessing functions used during training
    input_df = create_date_features(input_df)    # This creates year, month from Tranc_YearMonth
    input_df = create_age_features(input_df)     # This creates building_age, remaining_lease, lease_decay
    
    # Ensure proper data types based on JSON schema
    categorical_features = ["town", "flat_type", "storey_range", "flat_model", "market_hawker", 
                           "multistorey_carpark", "precinct_pavilion", "mrt_name", "bus_interchange", 
                           "mrt_interchange", "pri_sch_affiliation", "affiliation"]
    
    numerical_features = ["floor_area_sqm", "lease_commence_date", "hdb_age", "max_floor_lvl", 
                         "Mall_Nearest_Distance", "Hawker_Nearest_Distance", "mrt_nearest_distance", 
                         "bus_stop_nearest_distance", "pri_sch_nearest_distance", "sec_sch_nearest_dist", 
                         "cutoff_point", "year", "month", "building_age", "remaining_lease", "lease_decay"]
    
    # Convert data types
    for col in categorical_features:
        if col in input_df.columns:
            input_df[col] = input_df[col].astype(str)
    
    for col in numerical_features:
        if col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0.0).astype(float)
    
    # Ensure column order matches training data (all_features from JSON)
    expected_columns = ["town", "flat_type", "storey_range", "floor_area_sqm", "flat_model", 
                       "lease_commence_date", "hdb_age", "max_floor_lvl", "market_hawker", 
                       "multistorey_carpark", "precinct_pavilion", "Mall_Nearest_Distance", 
                       "Hawker_Nearest_Distance", "mrt_nearest_distance", "mrt_name", "bus_interchange", 
                       "mrt_interchange", "bus_stop_nearest_distance", "pri_sch_nearest_distance", 
                       "pri_sch_affiliation", "sec_sch_nearest_dist", "cutoff_point", "affiliation", 
                       "year", "month", "building_age", "remaining_lease", "lease_decay"]
    
    # Reorder columns to match training
    available_columns = [col for col in expected_columns if col in input_df.columns]
    input_df = input_df[available_columns]
    
    return input_df

def make_prediction(input_data, models_dict, model_type='ridge'):
    """Make a price prediction using the selected model."""
    try:
        # Check if model exists
        pipeline_key = f"{model_type}_pipeline"
        
        if pipeline_key in models_dict and models_dict[pipeline_key] is not None:
            # Get the pipeline
            pipeline = models_dict[pipeline_key]
            
            # Prepare input data
            input_df = prepare_features_for_prediction(input_data)
            
            # Debug section
            with st.expander("Debug Info", expanded=False):  
                st.markdown("### Input Data After Preparation")
                st.dataframe(input_df)
                
                # Show data types
                st.markdown("### Data Types")
                dtypes_df = pd.DataFrame({
                    'Column': input_df.columns,
                    'Data Type': [str(dtype) for dtype in input_df.dtypes],
                    'Sample Value': [str(input_df[col].iloc[0]) for col in input_df.columns]
                })
                st.dataframe(dtypes_df)
                
                # Check for problematic values
                st.markdown("### Potential Issues")
                for col in input_df.columns:
                    val = input_df[col].iloc[0]
                    if pd.isna(val):
                        st.error(f"❌ {col}: Contains NaN value")
                    elif str(val) in ['nan', 'None', 'inf', '-inf']:
                        st.error(f"❌ {col}: Contains problematic value '{val}'")
                    else:
                        st.success(f"✅ {col}: {type(val).__name__} = '{val}'")
            
            # Make prediction
            predicted_price = pipeline.predict(input_df)[0]
                
        else:
            return None, {"error": f"Model '{model_type}' not available"}
        
        # Get model metrics
        model_metrics = models_dict.get(f"{model_type}_metrics", {})
        if model_metrics is None:
            model_metrics = {}
        
        model_r2 = model_metrics.get('test_r2', 0)
        model_rmse = model_metrics.get('test_rmse', 0)
        
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
    """Display the prediction interface - ONLY collect base features from JSON."""
    # Header
    st.markdown("<h1 class='main-header'>Make Prediction</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    Enter property details below to get a predicted resale price. Only base features are collected - derived features are calculated automatically.
    """)
    
    # Load feature options
    feature_options = load_towns_and_features()
    
    # Ensure models are loaded
    if 'models' not in st.session_state:
        st.error("Models not loaded. Please refresh the application.")
        return
        
    models_dict = st.session_state['models']
    
    # Form with ONLY base features needed
    with st.form("prediction_form"):
        st.markdown("### Property Information")
        
        # Layout in 3 columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Location & Type**")
            
            town = st.selectbox(
                "Town",
                options=feature_options.get('town', ['ANG MO KIO']),
                index=0
            )
            
            flat_type = st.selectbox(
                "Flat Type",
                options=feature_options.get('flat_type', ['4 ROOM']),
                index=0
            )
            
            flat_model = st.selectbox(
                "Flat Model", 
                options=feature_options.get('flat_model', ['Improved']),
                index=0
            )
            
            storey_range = st.selectbox(
                "Storey Range",
                options=feature_options.get('storey_range', ['04 TO 06']),
                index=0
            )
        
        with col2:
            st.markdown("**Physical Attributes**")
            
            floor_area_sqm = st.slider(
                "Floor Area (sqm)",
                min_value=float(feature_options.get('floor_area_sqm_min', 28)),
                max_value=float(feature_options.get('floor_area_sqm_max', 186)),
                value=float(feature_options.get('floor_area_sqm_mean', 90)),
                step=1.0
            )
            
            lease_commence_date = st.slider(
                "Lease Commence Date",
                min_value=int(feature_options.get('lease_commence_date_min', 1960)),
                max_value=int(feature_options.get('lease_commence_date_max', 2020)),
                value=int(feature_options.get('lease_commence_date_mean', 1995)),
                step=1,
                help="Year when the flat's 99-year lease started"
            )
            
            max_floor_lvl = st.slider(
                "Max Floor Level",
                min_value=int(feature_options.get('max_floor_lvl_min', 3)),
                max_value=int(feature_options.get('max_floor_lvl_max', 50)),
                value=int(feature_options.get('max_floor_lvl_mean', 12)),
                step=1
            )
            
            # Boolean facilities (categorical in your model)
            market_hawker = st.selectbox("Market/Hawker in Block", options=['N', 'Y'], index=0)
            multistorey_carpark = st.selectbox("Multistorey Carpark", options=['N', 'Y'], index=1)
            precinct_pavilion = st.selectbox("Precinct Pavilion", options=['N', 'Y'], index=0)
        
        with col3:
            st.markdown("**Location & Transport**")
            
            Mall_Nearest_Distance = st.slider(
                "Distance to Nearest Mall (m)",
                min_value=float(feature_options.get('Mall_Nearest_Distance_min', 0)),
                max_value=float(feature_options.get('Mall_Nearest_Distance_max', 5000)),
                value=float(feature_options.get('Mall_Nearest_Distance_mean', 800)),
                step=50.0
            )
            
            Hawker_Nearest_Distance = st.slider(
                "Distance to Nearest Hawker (m)",
                min_value=float(feature_options.get('Hawker_Nearest_Distance_min', 0)),
                max_value=float(feature_options.get('Hawker_Nearest_Distance_max', 3000)),
                value=float(feature_options.get('Hawker_Nearest_Distance_mean', 300)),
                step=25.0
            )
            
            mrt_nearest_distance = st.slider(
                "Distance to Nearest MRT (m)",
                min_value=float(feature_options.get('mrt_nearest_distance_min', 0)),
                max_value=float(feature_options.get('mrt_nearest_distance_max', 5000)),
                value=float(feature_options.get('mrt_nearest_distance_mean', 600)),
                step=50.0
            )
            
            mrt_name = st.selectbox(
                "Nearest MRT Station",
                options=feature_options.get('mrt_name', ['Ang Mo Kio']),
                index=0
            )
        
        # Additional features
        st.markdown("### Additional Location Features")
        col4, col5, col6 = st.columns(3)
        
        with col4:
            bus_stop_nearest_distance = st.slider(
                "Distance to Nearest Bus Stop (m)",
                min_value=float(feature_options.get('bus_stop_nearest_distance_min', 0)),
                max_value=float(feature_options.get('bus_stop_nearest_distance_max', 1000)),
                value=float(feature_options.get('bus_stop_nearest_distance_mean', 150)),
                step=25.0
            )
            
            bus_interchange = st.selectbox("MRT is Bus Interchange", options=[0, 1], index=0)
            mrt_interchange = st.selectbox("MRT is Train Interchange", options=[0, 1], index=0)
        
        with col5:
            pri_sch_nearest_distance = st.slider(
                "Distance to Nearest Primary School (m)",
                min_value=float(feature_options.get('pri_sch_nearest_distance_min', 0)),
                max_value=float(feature_options.get('pri_sch_nearest_distance_max', 3000)),
                value=float(feature_options.get('pri_sch_nearest_distance_mean', 400)),
                step=25.0
            )
            
            pri_sch_affiliation = st.selectbox("Primary School Affiliation", options=[0, 1], index=0)
        
        with col6:
            sec_sch_nearest_dist = st.slider(
                "Distance to Nearest Secondary School (m)",
                min_value=float(feature_options.get('sec_sch_nearest_dist_min', 0)),
                max_value=float(feature_options.get('sec_sch_nearest_dist_max', 5000)),
                value=float(feature_options.get('sec_sch_nearest_dist_mean', 500)),
                step=25.0
            )
            
            cutoff_point = st.slider(
                "Secondary School PSLE Cutoff",
                min_value=int(feature_options.get('cutoff_point_min', 150)),
                max_value=int(feature_options.get('cutoff_point_max', 280)),
                value=int(feature_options.get('cutoff_point_mean', 200)),
                step=1
            )
            
            affiliation = st.selectbox("Secondary School Affiliation", options=[0, 1], index=0)
        
        # Transaction details
        st.markdown("### Transaction Details")
        col7, col8 = st.columns(2)
        
        with col7:
            transaction_date = st.date_input(
                "Transaction Date",
                value=datetime.now().date()
            )
        
        with col8:
            model_type = st.selectbox(
                "Prediction Model",
                options=["ridge", "linear", "lasso"],
                index=0,
                help="Select which model to use for prediction"
            )
        
        # Submit button
        submit = st.form_submit_button("Predict Price", use_container_width=True)
    
    # Make prediction when form is submitted
    if submit:
        # Calculate hdb_age from lease_commence_date
        current_year = datetime.now().year
        hdb_age = current_year - lease_commence_date
        
        # Collect ONLY the base input data (no derived features)
        input_data = {
            'town': town,
            'flat_type': flat_type,
            'flat_model': flat_model,
            'storey_range': storey_range,
            'floor_area_sqm': floor_area_sqm,
            'lease_commence_date': lease_commence_date,
            'hdb_age': hdb_age,  # Base calculation
            'max_floor_lvl': max_floor_lvl,
            'market_hawker': market_hawker,
            'multistorey_carpark': multistorey_carpark,
            'precinct_pavilion': precinct_pavilion,
            'Mall_Nearest_Distance': Mall_Nearest_Distance,
            'Hawker_Nearest_Distance': Hawker_Nearest_Distance,
            'mrt_nearest_distance': mrt_nearest_distance,
            'mrt_name': mrt_name,
            'bus_interchange': bus_interchange,
            'mrt_interchange': mrt_interchange,
            'bus_stop_nearest_distance': bus_stop_nearest_distance,
            'pri_sch_nearest_distance': pri_sch_nearest_distance,
            'pri_sch_affiliation': pri_sch_affiliation,
            'sec_sch_nearest_dist': sec_sch_nearest_dist,
            'cutoff_point': cutoff_point,
            'affiliation': affiliation,
            'year': transaction_date.year,
            'month': transaction_date.month
        }
        
        # Get the prediction (derived features will be calculated in prepare_features_for_prediction)
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
            
        else:
            st.error(f"Could not make prediction: {info.get('error', 'Unknown error')}")