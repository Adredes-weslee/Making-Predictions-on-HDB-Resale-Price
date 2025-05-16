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

The application uses a modular structure with separate modules for each page and
reusable components to maintain code organization and readability.

Typical usage:
    $ streamlit run app/main.py
"""
import os
import streamlit as st
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import page components
from app.pages.home import show_home
from app.pages.data_explorer import show_data_explorer
from app.pages.prediction import show_prediction
from app.pages.model_insights import show_model_insights

# Import UI components
from app.components.sidebar import create_sidebar

# Set page configuration
st.set_page_config(
    page_title="HDB Resale Price Prediction",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define the base directory
BASE_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    """Main function to run the Streamlit application.
    
    This function serves as the entry point for the application. It:
    1. Sets up the sidebar navigation using create_sidebar()
    2. Gets the user's selected page from the sidebar
    3. Routes to the appropriate page function based on the selection
    
    Each page function (show_home, show_data_explorer, etc.) is responsible for
    rendering its own content. This routing pattern keeps the main function clean
    and focused on application flow control.
    
    Returns:
        None: This function renders Streamlit UI directly and doesn't return values.
        
    Example:
        >>> main()
        # Renders the complete Streamlit application with navigation
    """
    # Create sidebar and get selected page
    page = create_sidebar(BASE_DIR)
    
    # Display appropriate page based on selection
    if page == "Home":
        show_home()
    elif page == "Data Explorer":
        show_data_explorer()
    elif page == "Make Prediction":
        show_prediction()
    elif page == "Model Insights":
        show_model_insights()


if __name__ == "__main__":
    main()
