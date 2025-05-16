"""Navigation sidebar component for the Streamlit application.

This module defines the sidebar UI component used throughout the HDB resale price 
prediction application. The sidebar provides consistent navigation across all pages
and includes the application title, logo (if available), page navigation menu, 
informational text, and version information.

The sidebar is designed to maintain a consistent look and feel across the entire
application while providing intuitive navigation between different sections.

Typical usage:
    >>> import streamlit as st
    >>> from pathlib import Path 
    >>> from app.components.sidebar import create_sidebar
    >>> base_dir = Path(__file__).parent.parent
    >>> current_page = create_sidebar(base_dir)
    >>> if current_page == "Home":
    ...     # Show home page content
"""
import os
import streamlit as st
from pathlib import Path

def create_sidebar(base_dir):
    """Create and configure the navigation sidebar for the application.
    
    This function builds the complete sidebar UI for the application, including:
    1. Application title and logo (if available)
    2. Navigation menu with page options
    3. Informational text about the application
    4. Footer with version information
    
    The function takes care of checking if assets like logos exist before
    attempting to load them, providing graceful fallback when assets are missing.
    
    Args:
        base_dir (Path): The base directory of the application, used to locate
            assets like the logo image.
    
    Returns:
        str: The name of the page selected by the user from the navigation menu.
            This should be used by the main app to determine which page to display.
    
    Example:
        >>> base_dir = Path(__file__).parent.parent
        >>> selected_page = create_sidebar(base_dir)
        >>> if selected_page == "Make Prediction":
        ...     show_prediction_page()
    """
    # Add title and logo
    st.sidebar.title("HDB Resale Price Prediction")
    
    # Try to load the logo if it exists
    logo_path = os.path.join(base_dir, "assets", "hdb_logo.png")
    if os.path.exists(logo_path):
        st.sidebar.image(logo_path, width=150)
    
    # Navigation menu
    page = st.sidebar.radio(
        "Navigation",
        options=["Home", "Data Explorer", "Make Prediction", "Model Insights"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "This application provides insights into HDB resale prices and "
        "allows you to predict the price of a flat based on various features."
    )
    
    # Add footer with version info
    st.sidebar.markdown("---")
    st.sidebar.caption("HDB Price Prediction App v1.0.0")
    
    return page
