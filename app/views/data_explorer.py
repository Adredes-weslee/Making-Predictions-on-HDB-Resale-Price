"""Data explorer component for the Streamlit application.

This module provides the Data Explorer page for the HDB resale price prediction
Streamlit application. It allows users to interactively explore and visualize the
HDB resale transaction dataset through various charts, maps, and statistical summaries.

The module includes functions for:
1. Loading and preparing the dataset for exploration
2. Displaying an overview of the data with summary statistics
3. Visualizing price distributions across different dimensions
4. Analyzing geographical price patterns with interactive maps
5. Examining relationships between features and prices
6. Exploring temporal trends in the resale market

Each visualization is designed to provide insights into different aspects of the
Singapore HDB resale market, helping users understand the factors that influence
property prices.

Typical usage:
    >>> import streamlit as st
    >>> from app.pages.data_explorer import show_data_explorer
    >>> show_data_explorer()
"""
import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import utility functions
from src.data.loader import get_data_paths, load_raw_data
from src.visualization.plots import (
    set_plotting_style, 
    plot_price_distribution, 
    plot_price_by_town,
    plot_price_by_flat_type,
    plot_price_trend,
    plot_correlation_heatmap,
    create_interactive_scatter,
    create_interactive_boxplot
)
from src.visualization.maps import create_singapore_map


def prepare_data():
    """Load and prepare HDB resale data for visualization in the explorer.
    
    This function handles the data acquisition and preparation process for the
    Data Explorer page. It:
    1. Loads the processed data file using the data loader functions
    2. Falls back to raw training data if processed data isn't available
    3. Converts date-related columns to proper datetime format
    4. Adds derived features like flat age if they don't already exist
    5. Performs any necessary data cleaning specific to visualization needs
    
    The function is designed to handle exceptions gracefully, allowing the
    application to provide useful feedback when data loading fails.
    
    Returns:
        pd.DataFrame: Prepared dataset with all necessary columns for visualizations.
            Contains standard columns like 'town', 'flat_type', 'floor_area_sqm',
            'resale_price', etc., plus any derived features.
            
    Raises:
        FileNotFoundError: If neither processed nor raw data files are found.
            This is caught internally in the show_data_explorer function.
            
    Example:
        >>> df = prepare_data()
        >>> print(f"Loaded {len(df)} HDB resale transactions")
    """
    # Load data
    data_paths = get_data_paths()
    try:
        # Access the processed csv file directly instead of the directory
        processed_dir = data_paths["processed"] 
        processed_file = os.path.join(processed_dir, "train_processed.csv")
        df = load_raw_data(processed_file)
    except FileNotFoundError:
        # Fallback to train data if processed not available
        df = load_raw_data(data_paths["train"])
    
    # Convert time-related columns if necessary
    if 'Tranc_YearMonth' in df.columns:
        df['Tranc_YearMonth'] = pd.to_datetime(df['Tranc_YearMonth'])
    
    # Add derived features if they don't exist
    if 'flat_age' not in df.columns and 'lease_commence_date' in df.columns:
        current_year = pd.to_datetime('today').year
        df['flat_age'] = current_year - df['lease_commence_date']
    
    return df


def show_data_explorer():
    """Display the data explorer page of the application.
    
    This is the main entry point for the Data Explorer page. It:
    1. Sets up the page title and introduction
    2. Attempts to load and prepare the dataset
    3. Creates tabs for different categories of visualizations
    4. Handles data loading errors with appropriate user feedback
    
    The function uses a tab-based interface to organize visualizations into
    logical categories, allowing users to explore different aspects of the data.
    Each tab contains related visualizations and interactive controls that update
    the displayed information based on user selections.
    
    Args:
        None
        
    Returns:
        None: This function renders content directly to the Streamlit UI.
        
    Example:
        >>> import streamlit as st
        >>> from app.pages.data_explorer import show_data_explorer
        >>> # Include this in a Streamlit page selection
        >>> if page == "Data Explorer":
        >>>     show_data_explorer()
    """
    st.title("HDB Resale Data Explorer")
    
    try:
        # Load and prepare data
        df = prepare_data()
        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Data Overview", 
            "Price Distribution", 
            "Location Analysis", 
            "Temporal Trends",
            "Feature Relationships"
        ])
        
        with tab1:
            show_data_overview(df)
            
        with tab2:
            show_price_distribution(df)
            
        with tab3:
            show_location_analysis(df)
            
        with tab4:
            show_temporal_trends(df)
            
        with tab5:
            show_feature_relationships(df)
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.warning("Please check that the data files are available in the correct location.")


def show_data_overview(df):
    """Show basic data overview with summary statistics and column information.
    
    This function presents a comprehensive overview of the dataset including:
    1. Basic dataset dimensions (number of records and features)
    2. A preview of the first few rows of data
    3. Descriptions of each column and its purpose in the dataset
    4. Summary statistics for key numerical columns
    5. Value counts for important categorical columns
    
    The overview provides users with a fundamental understanding of the dataset
    before they explore more detailed visualizations.
    
    Args:
        df (pd.DataFrame): HDB resale dataset to analyze and present.
            Expected to contain standard columns like 'town', 'flat_type',
            'floor_area_sqm', 'resale_price', etc.
            
    Returns:
        None: This function renders Streamlit UI elements directly.
        
    Example:
        >>> df = prepare_data()
        >>> show_data_overview(df)
        # Displays dataset overview section in the Streamlit app
    """
    st.header("Data Overview")
    
    # Display basic info
    st.write(f"Dataset contains {df.shape[0]:,} records with {df.shape[1]} features.")
    
    # Data preview
    st.subheader("Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Column descriptions
    st.subheader("Column Descriptions")
    
    # Define column descriptions
    column_desc = {
        "resale_price": "The final transaction price of the HDB flat",
        "town": "The town/estate where the flat is located",
        "flat_type": "The type/size of the flat (e.g., 3 ROOM, 4 ROOM)",
        "block": "The block number of the flat",
        "street_name": "Street name where the flat is located",
        "storey_range": "The range of storeys that the flat is located on",
        "floor_area_sqm": "The floor area of the flat in square meters",
        "flat_model": "The model of the flat (e.g., Improved, New Generation)",
        "lease_commence_date": "The year when the flat's lease commenced",
        "remaining_lease": "The remaining lease of the flat at time of transaction",
        "Tranc_YearMonth": "The year and month when the transaction occurred"
    }
    
    # Display column descriptions as a dataframe
    desc_df = pd.DataFrame(
        {"Description": [column_desc.get(col, "No description available") for col in df.columns]},
        index=df.columns
    )
    st.dataframe(desc_df, use_container_width=True)
    
    # Data statistics
    st.subheader("Numerical Features Statistics")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    st.dataframe(df[numeric_cols].describe().T, use_container_width=True)
    
    # Categorical data distribution
    st.subheader("Categorical Features Distribution")
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'town' in df.columns:
            st.write("#### Top Towns by Frequency")
            st.dataframe(df['town'].value_counts().head(10), use_container_width=True)
    
    with col2:
        if 'flat_type' in df.columns:
            st.write("#### Flat Types Distribution")
            st.dataframe(df['flat_type'].value_counts(), use_container_width=True)


def show_price_distribution(df):
    """Show visualizations of HDB resale price distributions.
    
    This function creates a comprehensive set of visualizations showing how
    HDB resale prices are distributed:
    1. Overall price distribution with histogram and KDE
    2. Price distribution by year to show trends over time
    3. Boxplots of prices grouped by relevant categorical variables
    4. Statistical summaries of the price distribution
    
    The visualizations help users understand the central tendency, spread,
    and potential outliers in the price data, as well as how prices vary
    across different property attributes.
    
    Args:
        df (pd.DataFrame): HDB resale dataset to analyze.
            Must contain 'resale_price' column and preferably categorical
            columns like 'flat_type', 'town', etc.
            
    Returns:
        None: This function renders Streamlit UI elements directly.
        
    Example:
        >>> df = prepare_data()
        >>> show_price_distribution(df)
        # Displays price distribution visualizations in the Streamlit app
    """
    st.header("Price Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Overall Price Distribution")
        fig = plot_price_distribution(df)
        st.pyplot(fig)
        
        # Summary statistics
        st.write("#### Price Summary Statistics")
        price_stats = {
            "Mean": df['resale_price'].mean(),
            "Median": df['resale_price'].median(),
            "Standard Deviation": df['resale_price'].std(),
            "Minimum": df['resale_price'].min(),
            "Maximum": df['resale_price'].max(),
            "Range": df['resale_price'].max() - df['resale_price'].min()
        }
        
        stats_df = pd.DataFrame({"Value": price_stats}).transpose()
        st.dataframe(stats_df, use_container_width=True)
    
    with col2:
        st.write("### Price Distribution by Flat Type")
        fig = plot_price_by_flat_type(df)
        st.pyplot(fig)
    
    # Interactive flat type comparison
    st.write("### Interactive Price Comparison by Flat Type")
    st.plotly_chart(create_interactive_boxplot(df, 'flat_type'), use_container_width=True)
    
    # Price by town
    st.write("### Top Towns by Resale Price")
    
    n_towns = st.slider("Select number of towns to display", min_value=5, max_value=25, value=15)
    fig = plot_price_by_town(df, top_n=n_towns)
    st.pyplot(fig)


def show_location_analysis(df):
    """Show geospatial analysis of HDB resale prices across Singapore.
    
    This function creates interactive maps and geographical visualizations showing
    how HDB resale prices vary across different locations in Singapore:
    1. Interactive map of Singapore with property locations colored by price
    2. Bar charts comparing average prices across towns/neighborhoods
    3. Heatmaps showing price density across different areas
    4. Town-level aggregated statistics and comparisons
    
    These visualizations help users understand the geographical factors that
    influence property prices and identify price trends in different regions.
    
    Args:
        df (pd.DataFrame): HDB resale dataset to analyze.
            Must contain 'town' column and ideally 'Latitude' and 'Longitude'
            columns for mapping.
            
    Returns:
        None: This function renders Streamlit UI elements directly.
        
    Example:
        >>> df = prepare_data()
        >>> show_location_analysis(df)
        # Displays location-based visualizations in the Streamlit app
    """
    st.header("Location-based Analysis")
    
    # Check if location data is available
    has_coords = all(col in df.columns for col in ['Latitude', 'Longitude'])
    
    if has_coords:
        # Map visualization
        st.write("### HDB Resale Prices by Location")
        st.write("Each point represents an HDB flat. Hover over points to see details.")
        
        color_by = st.selectbox(
            "Color points by:",
            options=['resale_price', 'flat_type', 'floor_area_sqm', 'lease_commence_date'],
            index=0
        )
        
        map_fig = create_singapore_map(df, color_col=color_by)
        st.plotly_chart(map_fig, use_container_width=True)
    else:
        st.warning("Geographic coordinates (Latitude/Longitude) are not available in the dataset. "
                  "Using alternative location visualizations.")
    
    # Town-based analysis
    st.write("### Town-based Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("#### Average Price by Town")
        town_avg = df.groupby('town')['resale_price'].mean().sort_values(ascending=False)
        st.dataframe(town_avg, use_container_width=True)
    
    with col2:
        st.write("#### Average Floor Area by Town")
        town_area = df.groupby('town')['floor_area_sqm'].mean().sort_values(ascending=False)
        st.dataframe(town_area, use_container_width=True)
    
    # Town comparison
    st.write("### Town Comparison")
    selected_towns = st.multiselect(
        "Select towns to compare:",
        options=sorted(df['town'].unique()),
        default=df['town'].value_counts().head(5).index.tolist()
    )
    
    if selected_towns:
        town_df = df[df['town'].isin(selected_towns)]
        st.plotly_chart(create_interactive_boxplot(town_df, 'town'), use_container_width=True)


def show_temporal_trends(df):
    """Display visualizations showing how HDB resale prices change over time.
    
    This function generates various time-based visualizations to analyze the trends,
    seasonality, and cyclical patterns in the HDB resale market. It includes:
    1. A matplotlib line plot showing the overall price trend over time
    2. An interactive Streamlit line chart for exploring monthly price averages
    3. A bar chart for year-on-year comparison of average prices
    
    The function checks for the presence of time-related data (Tranc_YearMonth column)
    and displays an appropriate warning if this data is missing.
    
    Args:
        df (pd.DataFrame): DataFrame containing HDB resale transaction data,
            including a 'Tranc_YearMonth' column with transaction dates in
            datetime format and a 'resale_price' column with transaction prices.
        
    Returns:
        None: This function renders content directly to the Streamlit UI.
        
    Note:
        This function assumes that date preprocessing has already been applied
        to convert 'Tranc_YearMonth' to proper datetime format.
        
    Example:
        >>> import pandas as pd
        >>> import streamlit as st
        >>> from app.pages.data_explorer import show_temporal_trends
        >>> # Prepare data with datetime column
        >>> df = pd.read_csv('data/processed/resale_data.csv')
        >>> df['Tranc_YearMonth'] = pd.to_datetime(df['month'])
        >>> # Display temporal trends
        >>> show_temporal_trends(df)
    """
    st.header("Temporal Trends Analysis")
    
    # Check if time data is available
    has_time = 'Tranc_YearMonth' in df.columns
    
    if has_time:
        # Price trend over time
        st.write("### Price Trend Over Time")
        fig = plot_price_trend(df)
        st.pyplot(fig)
        
        # Interactive time series exploration
        st.write("### Interactive Time Series Exploration")
        
        # Group by month
        df_monthly = df.copy()
        df_monthly['Month'] = df_monthly['Tranc_YearMonth'].dt.to_period('M')
        monthly_avg = df_monthly.groupby('Month').agg({
            'resale_price': 'mean',
            'floor_area_sqm': 'mean'
        }).reset_index()
        monthly_avg['Month'] = monthly_avg['Month'].astype(str)
        
        # Plot
        st.line_chart(monthly_avg.set_index('Month')['resale_price'], use_container_width=True)
        
        # Year-on-year comparison
        st.write("### Year-on-Year Comparison")
        df_copy = df.copy()
        df_copy['year'] = df_copy['Tranc_YearMonth'].dt.year
        df_copy['month'] = df_copy['Tranc_YearMonth'].dt.month
        
        yearly_avg = df_copy.groupby('year')['resale_price'].mean().reset_index()
        st.bar_chart(yearly_avg.set_index('year'), use_container_width=True)
    else:
        st.warning("Time-related data is not available in the dataset. Cannot display temporal trends.")
        
        # Fallback to lease information if available
        if 'lease_commence_date' in df.columns:
            st.write("### Analysis by Lease Commencement Date")
            
            # Group by lease commence date
            lease_avg = df.groupby('lease_commence_date')['resale_price'].mean().reset_index()
            st.line_chart(lease_avg.set_index('lease_commence_date'), use_container_width=True)


def show_feature_relationships(df):
    """Display visualizations showing relationships between different features in the dataset.
    
    This function generates interactive visualizations to explore relationships
    between different features in the HDB resale dataset, with a particular focus
    on how various features relate to resale price. It includes:
    1. A correlation heatmap showing the strength of relationships between selected features
    2. An interactive scatter plot for exploring relationships between any two features
       with optional color coding by a third feature
    3. Side-by-side distribution plots for comparing feature distributions
    
    The visualizations include interactive elements that allow users to select
    which features to analyze, helping them discover patterns and relationships
    in the data that may inform predictive modeling.
    
    Args:
        df (pd.DataFrame): DataFrame containing HDB resale transaction data with
            various features including at least 'resale_price' and 'floor_area_sqm'.
            Should contain a mix of numerical and categorical features for analysis.
        
    Returns:
        None: This function renders content directly to the Streamlit UI.
        
    Raises:
        KeyError: If essential columns like 'resale_price' are missing from the DataFrame.
        
    Example:
        >>> import pandas as pd
        >>> import streamlit as st
        >>> from app.pages.data_explorer import show_feature_relationships
        >>> # Load data
        >>> df = pd.read_csv('data/processed/resale_data.csv')
        >>> # Display feature relationship analysis
        >>> show_feature_relationships(df)
    """
    st.header("Feature Relationships Analysis")
    
    # Select numerical features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Feature correlation
    st.write("### Feature Correlation Analysis")
    
    # Select features for correlation
    corr_features = st.multiselect(
        "Select features for correlation analysis:",
        options=numeric_cols,
        default=['resale_price', 'floor_area_sqm', 'lease_commence_date']
    )
    
    if len(corr_features) > 1:
        fig = plot_correlation_heatmap(df, corr_features)
        st.pyplot(fig)
    else:
        st.warning("Please select at least two features for correlation analysis.")
    
    # Scatter plot analysis
    st.write("### Feature Relationship Explorer")
    
    col1, col2 = st.columns(2)
    
    with col1:
        x_feature = st.selectbox(
            "Select feature for X-axis:",
            options=[col for col in numeric_cols if col != 'resale_price'],
            index=0
        )
    
    with col2:
        color_feature = st.selectbox(
            "Color by (optional):",
            options=['None'] + [col for col in df.columns if df[col].nunique() < 15],
            index=0
        )
    
    color_col = None if color_feature == 'None' else color_feature
    
    # Create interactive scatter plot
    scatter_fig = create_interactive_scatter(
        df, 
        x_col=x_feature, 
        color_col=color_col,
        hover_data=['town', 'flat_type', 'floor_area_sqm']
    )
    st.plotly_chart(scatter_fig, use_container_width=True)
    
    # Feature distribution comparison
    st.write("### Feature Distribution Comparison")
    
    feature_to_plot = st.selectbox(
        "Select feature to analyze distribution:",
        options=[col for col in numeric_cols if col != 'resale_price'],
        index=0
    )
    
    # Plot histograms side by side
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot for selected feature
    sns.histplot(df[feature_to_plot], kde=True, ax=ax[0])
    ax[0].set_title(f'Distribution of {feature_to_plot}')
    
    # Plot for price
    sns.histplot(df['resale_price'], kde=True, ax=ax[1])
    ax[1].set_title('Distribution of resale_price')
    
    st.pyplot(fig)
