"""Geospatial visualization functions for HDB resale data.

This module provides specialized functions for creating geospatial visualizations
of HDB resale property data across Singapore. It includes functions for generating
interactive maps, choropleth visualizations, and other geographic representations
that help analyze how property attributes and prices vary by location.

The visualizations in this module are built primarily with Plotly's mapping
capabilities, providing interactive features like zooming, panning, hover information,
and layer toggling that enhance exploratory data analysis.

The module focuses on different types of geographic analysis:
1. Point maps showing individual HDB transactions
2. Choropleth maps showing aggregated statistics by town/region
3. Comparative maps that display multiple variables simultaneously
4. Heat maps showing density of transactions or price hotspots

Typical usage:
    >>> import pandas as pd
    >>> from src.visualization.maps import create_singapore_map
    >>> df = pd.read_csv('data/processed/geo_enhanced_data.csv')
    >>> fig = create_singapore_map(df, color_col='resale_price')
    >>> fig.show()
"""
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_singapore_map(
    df: pd.DataFrame,
    color_col: str = 'resale_price',
    hover_data: Optional[List[str]] = None,
    title: str = 'Singapore HDB Resale Prices'
) -> go.Figure:
    """
    Create an interactive map of Singapore displaying HDB resale locations.

    This function generates a scatter mapbox visualization using Plotly, where each
    point represents an HDB resale transaction. The points are colored based on the
    specified column, allowing for visual analysis of trends such as price distribution
    or other attributes.

    Args:
        df (pd.DataFrame): A pandas DataFrame containing the data to visualize. It must
            include 'Latitude' and 'Longitude' columns for geographic coordinates.
        color_col (str): The name of the column in the DataFrame to use for coloring
            the points. Default is 'resale_price'.
        hover_data (List[str], optional): A list of column names to display additional
            information when hovering over a point. Default is ['town', 'flat_type', 'floor_area_sqm'].
        title (str): The title of the map. Default is 'Singapore HDB Resale Prices'.

    Returns:
        go.Figure: A Plotly Figure object representing the map visualization.

    Example:
        >>> import pandas as pd
        >>> from src.visualization.maps import create_singapore_map
        >>> df = pd.read_csv('data/processed/geo_enhanced_data.csv')
        >>> fig = create_singapore_map(df, color_col='resale_price')
        >>> fig.show()
    """
    if hover_data is None:
        hover_data = ['town', 'flat_type', 'floor_area_sqm']
    
    fig = px.scatter_mapbox(
        df,
        lat='Latitude',
        lon='Longitude',
        color=color_col,
        size_max=15,
        zoom=10.5,
        center={"lat": 1.35, "lon": 103.82},  # Singapore center
        hover_data=hover_data,
        title=title,
        mapbox_style="carto-positron",
        color_continuous_scale=px.colors.sequential.Viridis
    )
    
    fig.update_layout(
        height=700,
        width=1000,
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
    )
    
    return fig


def create_town_choropleth(
    df: pd.DataFrame,
    town_boundaries: Union[str, Dict],
    agg_col: str = 'resale_price',
    agg_func: str = 'median',
    title: str = 'Median Resale Prices by Town'
) -> go.Figure:
    """Create a choropleth map showing aggregated statistics by town in Singapore.
    
    This function generates an interactive choropleth map that visualizes aggregated
    HDB resale property statistics by town across Singapore. Each town is colored
    according to the aggregated value (e.g., median price), providing an intuitive
    geographic view of how property attributes vary by location.
    
    The function requires geojson town boundary data to properly render the town
    polygons on the map. This can be provided either as a file path or a pre-loaded
    dictionary.
    
    Args:
        df (pd.DataFrame): DataFrame containing HDB resale transaction data with a
            'town' column that matches the region identifiers in the town_boundaries
            geojson data.
        town_boundaries (Union[str, Dict]): Either a file path to a geojson file
            containing Singapore town boundary data, or a pre-loaded geojson
            dictionary object.
        agg_col (str, optional): The column to aggregate by town. Defaults to
            'resale_price'.
        agg_func (str, optional): The aggregation function to apply. Supported values:
            'median', 'mean', 'count', 'min', 'max', 'sum'. Defaults to 'median'.
        title (str, optional): The title for the choropleth map. Defaults to
            'Median Resale Prices by Town'.
    
    Returns:
        go.Figure: A Plotly Figure object containing the interactive choropleth map
            that can be displayed in notebooks or web applications.
            
    Raises:
        KeyError: If the 'town' column is not in the DataFrame or if town names
            don't match between the DataFrame and geojson data.
        FileNotFoundError: If town_boundaries is a string but the file doesn't exist.
        TypeError: If town_boundaries is neither a valid geojson dict nor a string
            path to a geojson file.
        ValueError: If an unsupported agg_func is provided.
            
    Example:
        >>> import pandas as pd
        >>> import json
        >>> from src.visualization.maps import create_town_choropleth
        >>> # Load data
        >>> df = pd.read_csv('data/processed/resale_data.csv')
        >>> # Method 1: Pass geojson file path
        >>> fig = create_town_choropleth(df, 'data/geo/sg_town_boundaries.geojson')
        >>> # Method 2: Load geojson explicitly
        >>> with open('data/geo/sg_town_boundaries.geojson', 'r') as f:
        >>>     town_geo = json.load(f)
        >>> fig = create_town_choropleth(df, town_geo, agg_col='price_per_sqm', agg_func='mean')
        >>> fig.show()
    """
    # Aggregate data by town
    town_data = df.groupby('town')[agg_col].agg(agg_func).reset_index()
    
    # Create choropleth map
    fig = px.choropleth_mapbox(
        town_data,
        geojson=town_boundaries,
        locations='town',
        featureidkey='properties.name',
        color=agg_col,
        color_continuous_scale=px.colors.sequential.Viridis,
        mapbox_style="carto-positron",
        zoom=10.5,
        center={"lat": 1.35, "lon": 103.82},  # Singapore center
        opacity=0.7,
        title=title,
        labels={agg_col: f'{agg_func.title()} {agg_col.replace("_", " ").title()}'}
    )
    
    fig.update_layout(
        height=700,
        width=1000,
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
    )
    
    return fig


def create_comparative_town_maps(
    df: pd.DataFrame,
    color_col1: str = 'resale_price',
    color_col2: str = 'floor_area_sqm',
    agg_func: str = 'median'
) -> go.Figure:
    """Create a dual-map visualization comparing two metrics across Singapore towns.
    
    This function generates two side-by-side interactive maps of Singapore that allow
    simultaneous comparison of different attributes across towns. Each town is 
    represented by a point located at its geographical center, with colors indicating
    the aggregated value (e.g., median) of the selected metrics.
    
    The side-by-side presentation makes it easy to identify correlations or divergences
    between different metrics across geographic regions, such as how price relates to
    floor area or other housing attributes.
    
    Args:
        df (pd.DataFrame): DataFrame containing HDB resale transactions with at least
            'town', 'Latitude', and 'Longitude' columns. The DataFrame must also 
            contain the columns specified in color_col1 and color_col2.
        color_col1 (str, optional): Column name for the first metric to visualize.
            This will be used to color points in the first map.
            Defaults to 'resale_price'.
        color_col2 (str, optional): Column name for the second metric to visualize.
            This will be used to color points in the second map.
            Defaults to 'floor_area_sqm'.
        agg_func (str, optional): Aggregation function to apply to both metrics
            ('mean', 'median', 'count', 'sum', etc.).
            Defaults to 'median'.
        
    Returns:
        go.Figure: A Plotly Figure object containing two interactive maps side-by-side,
            that can be displayed in notebooks, saved to HTML, or rendered in Streamlit.
            
    Raises:
        KeyError: If 'town', 'Latitude', 'Longitude', or either of the specified
            color columns are not present in the DataFrame.
            
    Example:
        >>> import pandas as pd
        >>> from src.visualization.maps import create_comparative_town_maps
        >>> # Load data with geographical coordinates
        >>> df = pd.read_csv('data/processed/geo_enhanced_data.csv')
        >>> # Compare median resale prices and floor areas
        >>> fig = create_comparative_town_maps(df)
        >>> # Compare median prices and remaining lease years
        >>> fig2 = create_comparative_town_maps(
        ...     df,
        ...     color_col1='resale_price',
        ...     color_col2='remaining_lease',
        ...     agg_func='median'
        ... )
    """
    # Aggregate data by town
    town_data = df.groupby('town').agg({
        color_col1: agg_func,
        color_col2: agg_func,
        'Latitude': 'mean',
        'Longitude': 'mean'
    }).reset_index()
    
    # Create figure with two subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f'{agg_func.title()} {color_col1.replace("_", " ").title()} by Town',
            f'{agg_func.title()} {color_col2.replace("_", " ").title()} by Town'
        ],
        specs=[[{"type": "mapbox"}, {"type": "mapbox"}]],
        horizontal_spacing=0.05
    )
    
    # Add first map
    fig.add_trace(
        go.Scattermapbox(
            lat=town_data['Latitude'],
            lon=town_data['Longitude'],
            mode='markers',
            marker=dict(
                size=10,
                color=town_data[color_col1],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title=f'{color_col1.replace("_", " ").title()}',
                    x=0.46
                )
            ),
            text=town_data['town'],
            hoverinfo='text+z',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Add second map
    fig.add_trace(
        go.Scattermapbox(
            lat=town_data['Latitude'],
            lon=town_data['Longitude'],
            mode='markers',
            marker=dict(
                size=10,
                color=town_data[color_col2],
                colorscale='Plasma',
                showscale=True,
                colorbar=dict(
                    title=f'{color_col2.replace("_", " ").title()}',
                    x=1.0
                )
            ),
            text=town_data['town'],
            hoverinfo='text+z',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Update map settings
    fig.update_layout(
        height=600,
        width=1200,
        title_text=f"Comparison of {color_col1.replace('_', ' ').title()} and {color_col2.replace('_', ' ').title()} by Town",
        mapbox1=dict(
            center=dict(lat=1.35, lon=103.82),
            style="carto-positron",
            zoom=9.5
        ),
        mapbox2=dict(
            center=dict(lat=1.35, lon=103.82),
            style="carto-positron",
            zoom=9.5
        )
    )
    
    return fig
