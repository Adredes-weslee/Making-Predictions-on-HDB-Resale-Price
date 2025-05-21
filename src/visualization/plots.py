"""Core plotting functions for HDB resale data visualization.

This module provides a comprehensive collection of visualization functions for analyzing
and presenting HDB resale property data. It includes both static (Matplotlib/Seaborn)
and interactive (Plotly) visualization capabilities tailored to common real estate
data analysis tasks.

The module is organized around different visualization types:
1. Distribution plots - Showing the distribution of prices and other numerical features
2. Comparative plots - Comparing prices across different categories (towns, flat types)
3. Time series plots - Displaying trends in prices over time
4. Relationship plots - Exploring correlations between features
5. Interactive plots - Providing dynamic, explorable visualizations for deeper analysis

These visualization functions are used throughout the application to provide insights
into HDB resale property data patterns and support data-driven decision making.

Typical usage:
    >>> import pandas as pd
    >>> from src.visualization.plots import plot_price_distribution, set_plotting_style
    >>> 
    >>> # Load your data
    >>> df = pd.read_csv('data/processed/resale_data.csv')
    >>> 
    >>> # Set consistent plotting style
    >>> set_plotting_style()
    >>> 
    >>> # Create visualizations
    >>> fig = plot_price_distribution(df)
    >>> fig.savefig('outputs/price_distribution.png')
"""
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def set_plotting_style():
    """Set consistent styling for all static matplotlib visualizations.
    
    This function applies the FiveThirtyEight visualization style to all subsequent
    matplotlib plots, ensuring a consistent and professional appearance across
    all static visualizations in the application. It configures figure size, 
    font sizes for titles, labels, and tick marks to enhance readability.
    
    The style provides a clean, modern aesthetic with a light grid background
    and is optimized for data presentation in reports and dashboards.
    
    Args:
        None
        
    Returns:
        None: This function modifies the global matplotlib settings but doesn't
        return any value.
        
    Example:
        >>> from src.visualization.plots import set_plotting_style
        >>> # Apply consistent style to all subsequent plots
        >>> set_plotting_style()
        >>> # Create plots that will use this style
        >>> plt.figure()
        >>> plt.plot([1, 2, 3], [4, 5, 6])
        >>> plt.title('This plot uses the custom style')
    """
    plt.style.use('fivethirtyeight')
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12


def plot_price_distribution(df: pd.DataFrame) -> plt.Figure:
    """Create a histogram with KDE showing the distribution of HDB resale prices.
    
    This function visualizes the distribution of property resale prices using both
    a histogram and a kernel density estimation (KDE) curve. The plot helps identify
    the central tendency, spread, and skewness of the price distribution. It also
    displays the calculated skewness value as an annotation on the plot.
    
    Understanding the price distribution is fundamental for market analysis, helping
    to identify price ranges where most properties are traded and potential outliers
    in the data.
    
    Args:
        df (pd.DataFrame): DataFrame containing HDB resale transaction data with a
            'resale_price' column that contains numeric price values.
        
    Returns:
        plt.Figure: A Matplotlib Figure object containing the histogram and KDE plot
            that can be displayed in notebooks, saved to disk, or embedded in reports.
            
    Raises:
        KeyError: If 'resale_price' column is not present in the DataFrame.
        
    Example:
        >>> import pandas as pd
        >>> from src.visualization.plots import plot_price_distribution, set_plotting_style
        >>> # Load data and set style
        >>> df = pd.read_csv('data/processed/resale_data.csv')
        >>> set_plotting_style()
        >>> # Create and display price distribution plot
        >>> fig = plot_price_distribution(df)
        >>> fig.show()
        >>> # Save the figure to a file
        >>> fig.savefig('outputs/price_distribution.png', dpi=300, bbox_inches='tight')
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.histplot(df['resale_price'], kde=True, ax=ax)
    
    ax.set_title('Distribution of HDB Resale Prices')
    ax.set_xlabel('Resale Price (SGD)')
    ax.set_ylabel('Count')
    
    # Add skewness value as text
    skewness = df['resale_price'].skew()
    ax.text(
        0.95, 0.95, 
        f'Skewness: {skewness:.4f}',
        transform=ax.transAxes,
        ha='right',
        va='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    return fig


def plot_price_by_town(df: pd.DataFrame, top_n: int = 15) -> plt.Figure:
    """Create a boxplot showing resale price distributions across top Singapore towns.
    
    This function generates a horizontal boxplot that compares HDB resale price 
    distributions across the top N towns ranked by median price. The boxplot displays
    the median, quartiles, and outliers for each town, allowing for an intuitive
    comparison of price ranges and variability between different areas in Singapore.
    
    This visualization is useful for identifying premium property locations and
    understanding the price differentials between different geographic areas.
    
    Args:
        df (pd.DataFrame): DataFrame containing HDB resale transaction data with both
            'town' and 'resale_price' columns.
        top_n (int, optional): Number of top towns (by median price) to include in the
            visualization. Defaults to 15.
        
    Returns:
        plt.Figure: A Matplotlib Figure object containing the boxplot that can be
            displayed in notebooks, saved to disk, or embedded in reports.
            
    Raises:
        KeyError: If 'town' or 'resale_price' columns are not present in the DataFrame.
        ValueError: If top_n is not a positive integer.
        
    Example:
        >>> import pandas as pd
        >>> from src.visualization.plots import plot_price_by_town, set_plotting_style
        >>> # Load data and set style
        >>> df = pd.read_csv('data/processed/resale_data.csv')
        >>> set_plotting_style()
        >>> # Create boxplot with top 10 towns by median price
        >>> fig = plot_price_by_town(df, top_n=10)
        >>> # Display and save the figure
        >>> fig.show()
        >>> fig.savefig('outputs/price_by_town.png', dpi=300, bbox_inches='tight')
    """
    # Get top towns by median price
    town_median = df.groupby('town')['resale_price'].median().sort_values(ascending=False)
    top_towns = town_median.head(top_n).index.tolist()
    
    # Filter data
    df_filtered = df[df['town'].isin(top_towns)].copy()
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sns.boxplot(
        y='town',
        x='resale_price',
        data=df_filtered,
        ax=ax,
        order=top_towns
    )
    
    ax.set_title(f'Resale Prices by Town (Top {top_n})')
    ax.set_xlabel('Resale Price (SGD)')
    ax.set_ylabel('Town')
    
    # Format x-axis labels
    ax.ticklabel_format(style='plain', axis='x')
    
    return fig


def plot_price_by_flat_type(df: pd.DataFrame) -> plt.Figure:
    """Create a boxplot showing resale price distributions across different flat types.
    
    This function generates a horizontal boxplot comparing HDB resale price 
    distributions across different flat types (1 ROOM, 2 ROOM, etc.). The boxplot 
    displays the median, quartiles, and outliers for each flat type, ordered by size
    from smallest to largest.
    
    This visualization helps identify how property prices scale with flat size and
    type, revealing the price premium associated with larger unit types and the
    variability within each category.
    
    Args:
        df (pd.DataFrame): DataFrame containing HDB resale transaction data with both
            'flat_type' and 'resale_price' columns.
        
    Returns:
        plt.Figure: A Matplotlib Figure object containing the boxplot that can be
            displayed in notebooks, saved to disk, or embedded in reports.
            
    Raises:
        KeyError: If 'flat_type' or 'resale_price' columns are not present in the DataFrame.
        
    Example:
        >>> import pandas as pd
        >>> from src.visualization.plots import plot_price_by_flat_type, set_plotting_style
        >>> # Load data and set style
        >>> df = pd.read_csv('data/processed/resale_data.csv')
        >>> set_plotting_style()
        >>> # Create and display boxplot by flat type
        >>> fig = plot_price_by_flat_type(df)
        >>> fig.show()
        >>> # Save the figure to a file
        >>> fig.savefig('outputs/price_by_flat_type.png', dpi=300, bbox_inches='tight')
    """
    # Check if required columns exist and handle error gracefully
    if 'flat_type' not in df.columns or 'resale_price' not in df.columns:
        # Create a simple error message plot as fallback
        fig, ax = plt.subplots(figsize=(12, 6))
        missing_cols = []
        if 'flat_type' not in df.columns:
            missing_cols.append('flat_type')
        if 'resale_price' not in df.columns:
            missing_cols.append('resale_price')
        
        ax.text(0.5, 0.5, f"Cannot generate plot: Missing columns {', '.join(missing_cols)}",
                ha='center', va='center', fontsize=14)
        ax.set_axis_off()
        return fig
    
    # Check if flat_type column has values that can be used in our order list
    # and if there are enough unique values to make a meaningful boxplot
    if df['flat_type'].nunique() <= 1:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, "Not enough unique flat types to create a boxplot",
                ha='center', va='center', fontsize=14)
        ax.set_axis_off()
        return fig
    
    # Define flat type order - but only use values that exist in the data
    standard_order = ['1 ROOM', '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', 'EXECUTIVE', 'MULTI-GENERATION']
    
    # Filter to only include flat types that exist in our data
    existing_types = df['flat_type'].unique()
    flat_type_order = [ft for ft in standard_order if ft in existing_types]
    
    # If none of our standard types exist, just use whatever exists in the data
    if not flat_type_order:
        flat_type_order = sorted(existing_types)
    
    # Ensure we only plot data for flat types in our order list
    df_filtered = df[df['flat_type'].isin(flat_type_order)]
    
    # If after filtering we have no data, show an error message
    if len(df_filtered) == 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, "No data available for the defined flat types",
                ha='center', va='center', fontsize=14)
        ax.set_axis_off()
        return fig
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sns.boxplot(
        y='flat_type',
        x='resale_price',
        data=df_filtered,
        ax=ax,
        order=flat_type_order
    )
    
    ax.set_title('Resale Prices by Flat Type')
    ax.set_xlabel('Resale Price (SGD)')
    ax.set_ylabel('Flat Type')
    
    # Format x-axis labels
    ax.ticklabel_format(style='plain', axis='x')
    
    return fig


def plot_price_trend(df: pd.DataFrame) -> plt.Figure:
    """Create a time series plot showing the trend of mean HDB resale prices over time.
    
    This function generates a line plot visualizing how the average HDB resale prices
    have changed over time. The plot helps identify temporal patterns such as long-term
    trends, seasonality, and market cycles in the Singapore HDB resale market.
    
    Understanding price trends is essential for timing-related decisions, market
    forecasting, and identifying factors that may have influenced price movements
    at specific periods.
    
    Args:
        df (pd.DataFrame): DataFrame containing HDB resale transaction data with
            'Tranc_YearMonth' (datetime) and 'resale_price' columns. The datetime
            column should be in datetime format to allow proper temporal grouping.
        
    Returns:
        plt.Figure: A Matplotlib Figure object containing the time series plot that
            can be displayed in notebooks, saved to disk, or embedded in reports.
            
    Raises:
        KeyError: If 'Tranc_YearMonth' or 'resale_price' columns are not present
            in the DataFrame.
        TypeError: If 'Tranc_YearMonth' column is not in datetime format.
        
    Example:
        >>> import pandas as pd
        >>> from src.visualization.plots import plot_price_trend, set_plotting_style
        >>> # Load data and set style
        >>> df = pd.read_csv('data/processed/resale_data.csv')
        >>> df['Tranc_YearMonth'] = pd.to_datetime(df['month'])
        >>> set_plotting_style()
        >>> # Create and display time series plot
        >>> fig = plot_price_trend(df)
        >>> fig.show()
        >>> # Save the figure to a file
        >>> fig.savefig('outputs/price_trend.png', dpi=300, bbox_inches='tight')
    """
    # Group by transaction date
    df_grouped = df.groupby(df['Tranc_YearMonth'].dt.date).agg({'resale_price': 'mean'})
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(df_grouped.index, df_grouped['resale_price'])
    
    ax.set_title('Time Series of Mean HDB Resale Prices')
    ax.set_xlabel('Transaction Date')
    ax.set_ylabel('Mean Resale Price (SGD)')
    
    # Format y-axis labels
    ax.ticklabel_format(style='plain', axis='y')
    
    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    features: List[str],
    title: str = 'Correlation Heatmap'
) -> plt.Figure:
    """Create a correlation heatmap showing relationships between selected features.
    
    This function generates a triangular heatmap visualizing the Pearson correlation
    coefficients between selected numerical features. The heatmap uses a diverging
    color scale from blue (negative correlation) to red (positive correlation),
    with correlation values annotated on each cell. The triangular format avoids
    redundancy by showing each pairwise correlation only once.
    
    Correlation analysis helps identify which features are strongly related to each
    other, potentially revealing multicollinearity issues and suggesting feature
    importance for predictive modeling.
    
    Args:
        df (pd.DataFrame): DataFrame containing the numerical features to analyze.
        features (List[str]): List of column names to include in the correlation
            analysis. All columns must be numeric or convertible to numeric.
        title (str, optional): Title for the heatmap plot.
            Defaults to 'Correlation Heatmap'.
        
    Returns:
        plt.Figure: A Matplotlib Figure object containing the correlation heatmap
            that can be displayed in notebooks, saved to disk, or embedded in reports.
            
    Raises:
        KeyError: If any column name in features is not present in the DataFrame.
        TypeError: If any selected column cannot be converted to numeric data.
        
    Example:
        >>> import pandas as pd
        >>> from src.visualization.plots import plot_correlation_heatmap, set_plotting_style
        >>> # Load data and set style
        >>> df = pd.read_csv('data/processed/resale_data.csv')
        >>> set_plotting_style()
        >>> # Select numeric features for correlation analysis
        >>> features = ['resale_price', 'floor_area_sqm', 'remaining_lease', 'age_years']
        >>> # Create and display correlation heatmap
        >>> fig = plot_correlation_heatmap(df, features, 'HDB Price Correlations')
        >>> fig.show()
        >>> # Save the figure to a file
        >>> fig.savefig('outputs/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    """
    # Compute correlation matrix
    corr = df[features].corr()
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    mask = np.triu(np.ones_like(corr))
    
    sns.heatmap(
        corr,
        annot=True,
        cmap='coolwarm_r',
        vmin=-1,
        vmax=1,
        mask=mask,
        ax=ax
    )
    
    ax.set_title(title)
    plt.xticks(rotation=45, ha='right')
    
    return fig


def create_interactive_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str = 'resale_price',
    color_col: Optional[str] = None,
    hover_data: Optional[List[str]] = None
) -> go.Figure:
    """Create an interactive scatter plot with optional color grouping and hover data.
    
    This function generates an interactive Plotly scatter plot to visualize relationships
    between two variables, with options for color-coding by a third variable and 
    displaying additional information on hover. The scatter plot allows zooming, 
    panning, and hovering to explore data points in detail.
    
    Interactive scatter plots are particularly useful for exploring relationships
    between continuous variables and identifying patterns, clusters, outliers, and
    potential non-linear relationships in the data.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data to visualize.
        x_col (str): Column name for the x-axis variable.
        y_col (str, optional): Column name for the y-axis variable.
            Defaults to 'resale_price'.
        color_col (str, optional): Column name for color-coding the points. If provided,
            points will be colored according to their values in this column, which can 
            reveal additional patterns. If None, all points will have the same color.
            Defaults to None.
        hover_data (List[str], optional): List of additional column names to display
            when hovering over data points. If None, only x and y values are shown.
            Defaults to None.
        
    Returns:
        go.Figure: A Plotly Figure object containing the interactive scatter plot that
            can be displayed in notebooks, Dash applications, or exported to HTML.
            
    Raises:
        KeyError: If any of the specified column names are not present in the DataFrame.
        
    Example:
        >>> import pandas as pd
        >>> from src.visualization.plots import create_interactive_scatter
        >>> # Load data
        >>> df = pd.read_csv('data/processed/resale_data.csv')
        >>> # Create basic scatter plot of floor area vs price
        >>> fig = create_interactive_scatter(df, x_col='floor_area_sqm')
        >>> # Create enhanced scatter plot with color and hover data
        >>> fig2 = create_interactive_scatter(
        ...     df,
        ...     x_col='floor_area_sqm',
        ...     y_col='resale_price',
        ...     color_col='flat_type',
        ...     hover_data=['town', 'remaining_lease', 'storey_range']
        ... )
        >>> # Display the plot in a notebook or Streamlit app
        >>> fig2.show()
    """
    if hover_data is None:
        hover_data = []
    
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        hover_data=hover_data,
        title=f'{y_col} vs {x_col}',
        labels={
            x_col: x_col.replace('_', ' ').title(),
            y_col: y_col.replace('_', ' ').title()
        }
    )
    
    fig.update_layout(
        height=600,
        width=900,
        plot_bgcolor='rgba(240, 240, 240, 0.8)'
    )
    
    return fig


def create_interactive_boxplot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str = 'resale_price',
    color_col: Optional[str] = None
) -> go.Figure:
    """Create an interactive boxplot comparing distributions across categories.
    
    This function generates an interactive Plotly boxplot that compares the distribution
    of a numerical variable (typically 'resale_price') across different categories.
    Each box shows the median, quartiles, and outliers for its category, allowing for
    easy comparison of central tendency and spread. The plot supports interactive 
    features like zooming, panning, and hovering to explore the distributions in detail.
    
    Interactive boxplots are particularly useful for comparing price distributions
    across different categorical variables such as towns, flat types, or storey ranges,
    and identifying categories with higher medians or greater variability.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data to visualize.
        x_col (str): Column name for the categorical variable to group by on the x-axis.
        y_col (str, optional): Column name for the numerical variable to plot
            distributions of on the y-axis. Defaults to 'resale_price'.
        color_col (str, optional): Column name for an additional categorical variable
            to use for color grouping, creating nested boxplots. If None, no color
            grouping is applied. Defaults to None.
        
    Returns:
        go.Figure: A Plotly Figure object containing the interactive boxplot that
            can be displayed in notebooks, Dash applications, or exported to HTML.
            
    Raises:
        KeyError: If any of the specified column names are not present in the DataFrame.
        ValueError: If the x_col doesn't contain categorical data or y_col doesn't
            contain numerical data.
        
    Example:
        >>> import pandas as pd
        >>> from src.visualization.plots import create_interactive_boxplot
        >>> # Load data
        >>> df = pd.read_csv('data/processed/resale_data.csv')
        >>> # Create boxplot of prices by town
        >>> fig = create_interactive_boxplot(df, x_col='town')
        >>> # Create boxplot of prices by flat_type, colored by storey_range
        >>> fig2 = create_interactive_boxplot(
        ...     df,
        ...     x_col='flat_type',
        ...     y_col='resale_price',
        ...     color_col='storey_range'
        ... )
        >>> # Display the plot in a notebook or Streamlit app
        >>> fig2.show()
    """
    fig = px.box(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        title=f'{y_col} by {x_col}',
        labels={
            x_col: x_col.replace('_', ' ').title(),
            y_col: y_col.replace('_', ' ').title()
        }
    )
    
    fig.update_layout(
        height=600,
        width=900,
        plot_bgcolor='rgba(240, 240, 240, 0.8)'
    )
    
    return fig
