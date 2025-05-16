"""Reusable visualization components for the Streamlit application.

This module provides standardized, reusable visualization components specifically
designed for rendering in Streamlit. These components abstract common visualization
patterns used throughout the application, ensuring a consistent look and feel while
reducing code duplication.

The module includes functions for displaying summary metrics, distribution plots,
correlation matrices, geographic visualizations, and other data visualizations
relevant to HDB resale price analysis.

These components wrap lower-level visualization functions from src.visualization
with Streamlit-specific rendering and layout considerations.

Typical usage:
    >>> import streamlit as st
    >>> import pandas as pd
    >>> from app.components.visualizations import show_price_summary_metrics
    >>> df = pd.DataFrame({'resale_price': [300000, 400000, 500000]})
    >>> show_price_summary_metrics(df)
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

from src.visualization.plots import (
    set_plotting_style,
    plot_price_distribution,
    plot_price_by_town,
    plot_price_by_flat_type,
    plot_correlation_heatmap,
    create_interactive_scatter
)

from src.visualization.maps import create_singapore_map


def show_price_summary_metrics(df):
    """Display summary metrics for HDB resale price data in a row of cards.
    
    This function creates a row of four metric cards showing key statistics
    about the resale prices in the dataset: mean, median, minimum, and maximum.
    The values are formatted as Singapore dollars with commas and two decimal places.
    
    The function uses Streamlit's column layout and metric components for
    responsive design that works well on different screen sizes.
    
    Args:
        df (pd.DataFrame): DataFrame containing a 'resale_price' column with
            numerical values representing HDB resale prices.
    
    Returns:
        None: This function renders Streamlit components directly.
        
    Raises:
        KeyError: If the DataFrame does not contain a 'resale_price' column.
        
    Example:
        >>> df = pd.DataFrame({'resale_price': [300000, 400000, 500000]})
        >>> show_price_summary_metrics(df)
        # Displays four metric cards with price statistics
    """
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Average Price", f"S${df['resale_price'].mean():,.2f}")
    
    with col2:
        st.metric("Median Price", f"S${df['resale_price'].median():,.2f}")
    
    with col3:
        st.metric("Min Price", f"S${df['resale_price'].min():,.2f}")
    
    with col4:
        st.metric("Max Price", f"S${df['resale_price'].max():,.2f}")


def display_distribution_plot(df, column, title=None):
    """Display a histogram with KDE overlay for a specified dataframe column.
    
    This function creates a distribution plot (histogram with kernel density estimate)
    for a specified column in the provided DataFrame. The plot uses Seaborn's
    histplot function with a fixed figure size and styling consistent with
    the application's design language.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data to visualize.
        column (str): The name of the column to plot. Must exist in the DataFrame.
        title (str, optional): Custom title for the plot. If not provided, a default
            title will be generated using the column name.
    
    Returns:
        None: This function renders a Matplotlib figure directly to Streamlit.
        
    Raises:
        KeyError: If the specified column does not exist in the DataFrame.
        
    Example:
        >>> df = pd.DataFrame({'floor_area_sqm': [70, 75, 80, 85, 90]})
        >>> display_distribution_plot(df, 'floor_area_sqm', 'Floor Area Distribution')
        # Displays a histogram with density curve for floor area
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.histplot(df[column], kde=True, ax=ax)
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Distribution of {column}')
    
    ax.set_xlabel(column)
    ax.set_ylabel('Count')
    
    st.pyplot(fig)


def display_correlation_matrix(df, features=None):
    """Display an interactive correlation matrix heatmap for selected features.
    
    This function creates a correlation heatmap showing the Pearson correlation
    coefficients between selected numerical features in the dataset. The heatmap
    uses a diverging color scale to distinguish between positive and negative
    correlations, with annotations showing the exact correlation values.
    
    If no features are specified, the function automatically selects numerical
    features from the dataset, excluding any ID columns.
    
    Args:
        df (pd.DataFrame): DataFrame containing the features to analyze.
        features (list, optional): List of column names to include in the
            correlation matrix. If None, all numerical columns are used.
            Defaults to None.
    
    Returns:
        None: This function renders a heatmap directly to Streamlit.
        
    Raises:
        ValueError: If the specified features are not in the DataFrame or
            if less than two valid numerical features are available for correlation.
            
    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'price': [100, 200, 300],
        ...     'area': [50, 70, 90],
        ...     'age': [10, 5, 15]
        ... })
        >>> display_correlation_matrix(df, ['price', 'area', 'age'])
        # Displays a 3x3 correlation heatmap
    """
    if features is None:
        # Select only numeric columns
        features = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Compute correlation matrix
    corr = df[features].corr()
    
    # Create heatmap
    fig = plot_correlation_heatmap(df, features)
    st.pyplot(fig)


def create_map_with_controls(df):
    """Create an interactive map of HDB resale transactions with filter controls.
    
    This function generates a Streamlit component with an interactive map showing
    HDB resale transactions across Singapore, along with filter controls that allow
    users to:
    1. Filter by town/region
    2. Filter by flat type
    3. Set price range limits
    4. Choose which variable to use for color-coding points
    
    The map automatically updates when users change any filter settings, providing
    an interactive exploration experience.
    
    Args:
        df (pd.DataFrame): DataFrame containing HDB resale transactions with at least
            the columns 'town', 'flat_type', 'resale_price', 'Latitude', and 'Longitude'.
    
    Returns:
        None: This function renders Streamlit components directly.
        
    Raises:
        ValueError: If required geographical columns ('Latitude', 'Longitude') are
            missing from the DataFrame.
            
    Example:
        >>> import streamlit as st
        >>> from app.components.visualizations import create_map_with_controls
        >>> df = load_geo_enhanced_data()  # Function to load data with geo coordinates
        >>> create_map_with_controls(df)
        # Renders an interactive map with filter controls
    """
    # Check if location data exists
    if not all(col in df.columns for col in ['Latitude', 'Longitude']):
        st.warning("Geographic coordinates are not available in the dataset.")
        return
    
    # Controls for the map
    st.subheader("Map Controls")
    col1, col2 = st.columns(2)
    
    with col1:
        color_by = st.selectbox(
            "Color points by:",
            options=['resale_price', 'flat_type', 'floor_area_sqm', 'lease_commence_date'],
            index=0
        )
    
    with col2:
        # Filter by town if town column exists
        if 'town' in df.columns:
            selected_towns = st.multiselect(
                "Filter by town:",
                options=sorted(df['town'].unique()),
                default=[]
            )
            
            # Filter data if towns selected
            if selected_towns:
                filtered_df = df[df['town'].isin(selected_towns)]
            else:
                filtered_df = df
        else:
            filtered_df = df
    
    # Create map
    map_fig = create_singapore_map(filtered_df, color_col=color_by)
    st.plotly_chart(map_fig, use_container_width=True)


def display_price_comparisons(df, categorical_var='flat_type'):
    """Display comparative visualizations of prices across categorical variables.
    
    This function creates a set of visualizations showing how HDB resale prices
    vary across different categories (e.g., flat types, towns, storey ranges).
    It includes:
    1. Box plots showing price distributions within each category
    2. Bar charts showing average prices by category
    3. Count plots showing the number of transactions in each category
    4. Violin plots showing the detailed price distribution shape by category
    
    The visualizations help users understand both the central tendency and
    variability of prices across different categorical attributes.
    
    Args:
        df (pd.DataFrame): DataFrame containing HDB resale transactions with at
            least the columns 'resale_price' and the specified categorical variable.
        categorical_var (str, optional): The categorical variable to use for
            grouping and comparison. Defaults to 'flat_type'.
    
    Returns:
        None: This function renders Streamlit components directly.
        
    Raises:
        ValueError: If the specified categorical variable is not in the DataFrame.
            
    Example:
        >>> import streamlit as st
        >>> import pandas as pd
        >>> from app.components.visualizations import display_price_comparisons
        >>> df = pd.DataFrame({
        ...     'resale_price': [300000, 400000, 500000, 350000],
        ...     'flat_type': ['3 ROOM', '4 ROOM', '5 ROOM', '3 ROOM'],
        ...     'town': ['BEDOK', 'TAMPINES', 'BEDOK', 'ANG MO KIO']
        ... })
        >>> display_price_comparisons(df, 'town')
        # Displays price comparison visualizations grouped by town
    """
    # Group data
    grouped = df.groupby(categorical_var)['resale_price'].agg(['mean', 'median', 'count']).reset_index()
    grouped = grouped.sort_values('median', ascending=False)
    
    # Display grouped data
    st.dataframe(grouped, use_container_width=True)
    
    # Create interactive bar chart
    fig = px.bar(
        grouped, 
        x=categorical_var, 
        y='median',
        color='count',
        labels={'median': 'Median Price (SGD)', 'count': 'Number of Transactions'},
        title=f'Median Price by {categorical_var}',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
