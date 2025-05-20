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
    # Get configuration from session state
    app_config = st.session_state.get('app_config', {})
    
    # Get application settings from config
    app_title = app_config.get('app', {}).get('title', "HDB Resale Price Prediction")
    app_description = app_config.get('app', {}).get('description', 
        "This application provides insights into HDB resale prices and "
        "allows you to predict the price of a flat based on various features.")
    app_version = app_config.get('app', {}).get('version', "1.0.0")
    
    # Get page configuration
    pages_config = app_config.get('pages', {})
    available_pages = []
    page_titles = []
    
    # Build page navigation options from config
    if pages_config:
        for page_id, page_info in pages_config.items():
            if 'title' in page_info:
                available_pages.append(page_id)
                page_title = f"{page_info.get('icon', '')} {page_info.get('title')}"
                page_titles.append(page_title)
    else:
        # Fallback to default pages if config is missing
        available_pages = ["home", "data_explorer", "prediction", "model_insights"]
        page_titles = ["🏠 Home", "📊 Data Explorer", "🔮 Make Prediction", "📈 Model Insights"]
    
    # Add title and logo
    st.sidebar.title(app_title)
    
    # Try to load the logo if it exists
    logo_path = os.path.join(base_dir, "assets", "hdb_logo.png")
    if os.path.exists(logo_path):
        st.sidebar.image(logo_path, width=150)
    
    # Navigation menu
    selected_page_title = st.sidebar.radio(
        "Navigation",
        options=page_titles,
        index=0
    )
    
    # Convert the selected page title back to its ID
    selected_index = page_titles.index(selected_page_title)
    selected_page = available_pages[selected_index]
    
    # Map page IDs to the expected page names in main.py
    page_name_map = {
        "home": "Home",
        "data_explorer": "Data Explorer",
        "prediction": "Make Prediction",
        "model_insights": "Model Insights"
    }
    
    display_page = page_name_map.get(selected_page, selected_page.capitalize())
      # Add app description
    st.sidebar.markdown("---")
    st.sidebar.info(app_description)
    
    # Add footer with version info
    st.sidebar.markdown("---")
    st.sidebar.caption(f"HDB Price Prediction App v{app_version}")
    
    return display_page
