"""Main Streamlit application for HDB resale price prediction.

This is the entry point for the Streamlit web application. It orchestrates the navigation
and loads different page components based on user selection. The application follows a
multi-page architecture where the sidebar navigation controls which content is displayed
in the main area.

The application includes the following main sections:
1. Home - Introduction and overview of the project
2. Data Explorer - Interactive visualizations of HDB resale data
3. Make Prediction - Form for users to input property details and get price predictions
4. Model Insights - Analysis of model performance and feature importance

Typical usage:
    $ streamlit run app/main.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
from pathlib import Path
import logging

# Import views
from views.home import show_home
from views.data_explorer import show_data_explorer
from views.prediction import show_prediction
from views.model_insights import show_model_insights

# Import components
from components.sidebar import create_sidebar

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="HDB Resale Price Predictor",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Application title and styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load all pre-trained models at startup and cache them.
    
    Returns:
        dict: Dictionary containing loaded models and their metrics
    """
    try:
        # Get models directory
        root_dir = Path(__file__).parent.parent
        models_dir = os.path.join(root_dir, 'models')
        
        # Initialize models dictionary
        models = {}
        
        # Define model types to load
        model_types = ['linear', 'ridge', 'lasso']
        
        for model_type in model_types:
            try:
                # Load pipeline model
                model_path = os.path.join(models_dir, f"pipeline_{model_type}_model.pkl")
                if os.path.exists(model_path):
                    # Store with both keys for compatibility
                    pipeline = joblib.load(model_path)
                    models[f"{model_type}_pipeline"] = pipeline
                    models[f"{model_type}_model"] = pipeline
                    logger.info(f"Loaded {model_type} model successfully")
                    
                    # DEBUG: Print pipeline structure
                    if hasattr(pipeline, 'named_steps'):
                        logger.info(f"{model_type} pipeline steps: {list(pipeline.named_steps.keys())}")
                        
                        # Try different common step names
                        possible_step_names = ['regressor', 'classifier', 'model', model_type, 'estimator']
                        actual_model = None
                        
                        for step_name in possible_step_names:
                            if step_name in pipeline.named_steps:
                                actual_model = pipeline.named_steps[step_name]
                                logger.info(f"Found model in step '{step_name}' for {model_type}")
                                break
                        
                        # If no standard name found, try the last step
                        if actual_model is None:
                            step_names = list(pipeline.named_steps.keys())
                            if step_names:
                                last_step_name = step_names[-1]
                                actual_model = pipeline.named_steps[last_step_name]
                                logger.info(f"Using last step '{last_step_name}' as model for {model_type}")
                        
                        if actual_model is not None:
                            models[f"{model_type}_regressor"] = actual_model
                            logger.info(f"{model_type} regressor extracted. Has coef_: {hasattr(actual_model, 'coef_')}")
                            if hasattr(actual_model, 'coef_'):
                                logger.info(f"{model_type} coefficient shape: {actual_model.coef_.shape}")
                        else:
                            logger.warning(f"Could not extract regressor from {model_type} pipeline")
                            models[f"{model_type}_regressor"] = None
                    else:
                        # Pipeline doesn't have named_steps, might be the model itself
                        if hasattr(pipeline, 'coef_'):
                            models[f"{model_type}_regressor"] = pipeline
                            logger.info(f"{model_type} model is not a pipeline, using directly")
                        else:
                            logger.warning(f"{model_type} is neither a pipeline nor has coefficients")
                            models[f"{model_type}_regressor"] = None
                        
                else:
                    logger.warning(f"Model file not found: {model_path}")
                    models[f"{model_type}_pipeline"] = None
                    models[f"{model_type}_model"] = None
                    models[f"{model_type}_regressor"] = None
                
                # Load metrics
                metrics_path = os.path.join(models_dir, f"pipeline_{model_type}_model_metrics.json")
                if os.path.exists(metrics_path):
                    with open(metrics_path, 'r') as f:
                        models[f"{model_type}_metrics"] = json.load(f)
                else:
                    logger.warning(f"Metrics file not found: {metrics_path}")
                    models[f"{model_type}_metrics"] = None
                
                # Load feature info
                features_path = os.path.join(models_dir, f"pipeline_{model_type}_model_features.json")
                if os.path.exists(features_path):
                    with open(features_path, 'r') as f:
                        models[f"{model_type}_features"] = json.load(f)
                else:
                    logger.warning(f"Features file not found: {features_path}")
                    models[f"{model_type}_features"] = None
                    
            except Exception as e:
                logger.error(f"Error loading {model_type} model: {str(e)}")
                models[f"{model_type}_pipeline"] = None
                models[f"{model_type}_model"] = None
                models[f"{model_type}_regressor"] = None
                models[f"{model_type}_metrics"] = None
                models[f"{model_type}_features"] = None
        
        return models
    
    except Exception as e:
        logger.error(f"Error in load_models: {str(e)}")
        st.error(f"Failed to load models: {str(e)}")
        return {}
    
def main():
    """Main function to run the Streamlit app."""
    try:
        # Load models once at startup
        models = load_models()
        
        # Store models in session state for access across pages
        if 'models' not in st.session_state:
            st.session_state['models'] = models
        
        # Create sidebar and get selected page
        selected_page = create_sidebar()
        
        # Display selected page
        if selected_page == "Home":
            show_home()
        elif selected_page == "Data Explorer":
            show_data_explorer()
        elif selected_page == "Make Prediction":
            show_prediction()
        elif selected_page == "Model Performance":
            show_model_insights()
        else:
            show_home()  # Default to home
            
    except Exception as e:
        st.error(f"An error occurred in the application: {str(e)}")
        logger.error(f"Application error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()