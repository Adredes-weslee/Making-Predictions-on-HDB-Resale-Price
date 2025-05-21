"""Sidebar component for navigation and filters.

This module provides the sidebar component that allows users to navigate between
different sections of the application and apply global filters where applicable.
"""
import streamlit as st
import pandas as pd

def create_sidebar():
    """Create the sidebar navigation and filters.
    
    Returns:
        str: The selected page name
    """
    with st.sidebar:
        st.image("app/static/hdb_logo.png", width=120)
        st.title("HDB Resale Price Predictor")
        
        # Main navigation
        selected_page = st.radio(
            "Navigation",
            ["Home", "Data Explorer", "Make Prediction", "Model Performance"],
            index=0,  # Default to Home
            format_func=lambda x: f"📊 {x}" if x == "Data Explorer" else
                              f"🔮 {x}" if x == "Make Prediction" else
                              f"📈 {x}" if x == "Model Performance" else
                              f"🏠 {x}"
        )
        
        st.sidebar.markdown("---")
        
        # Add global filters if needed (town, flat type, etc.)
        if selected_page in ["Data Explorer", "Model Performance"]:
            st.subheader("Data Filters")
            
            # These filters could be used across multiple pages
            if 'towns' in st.session_state:
                towns = st.session_state.towns
                selected_towns = st.multiselect(
                    "Towns",
                    options=towns,
                    default=towns[:5] if len(towns) > 5 else towns
                )
                
                if selected_towns:
                    st.session_state['selected_towns'] = selected_towns
            
            if 'flat_types' in st.session_state:
                flat_types = st.session_state.flat_types
                selected_flat_types = st.multiselect(
                    "Flat Types",
                    options=flat_types,
                    default=flat_types
                )
                
                if selected_flat_types:
                    st.session_state['selected_flat_types'] = selected_flat_types
        
        # Information about the model
        st.sidebar.markdown("---")
        st.sidebar.info(
            """
            **About this app**  
            
            This app helps you explore HDB resale prices and make predictions based on property attributes.
            """
        )
        
        # Footer
        st.sidebar.markdown("---")
        st.sidebar.markdown("Developed by [Your Name]")
        st.sidebar.markdown("Data source: data.gov.sg")
        
    return selected_page