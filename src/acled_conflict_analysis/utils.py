"""
ACLED Conflict Analysis Utility Functions
==========================================

This module provides utility functions for data analysis and processing,
including quantile calculations for statistical binning.

Functions:
---------
- calculate_quantiles: Generic function to calculate any number of quantile bins
- calculate_quartiles: Calculate quartiles (4 bins) for data
- calculate_deciles: Calculate deciles (10 bins) for data
- calculate_terciles: Calculate terciles (3 bins) for data
"""

import pandas as pd
import numpy as np
from matplotlib.colors import Normalize


def calculate_quantiles(data, measure, n_quantiles, labels=None, return_norm=True):
    """
    Calculate quantile bin edges and assign quantile categories to the data.
    
    This function divides data into n equal-sized bins based on quantiles,
    ensuring approximately equal number of observations in each bin.
    
    Parameters:
    -----------
    data : pandas.DataFrame or geopandas.GeoDataFrame
        The dataset to calculate quantiles from.
    measure : str
        The numeric column name to base quantiles on.
    n_quantiles : int
        Number of quantile bins to create (e.g., 4 for quartiles, 10 for deciles).
    labels : list, optional
        Custom labels for the bins. If None, uses Q1, Q2, Q3, etc.
        Must have length equal to n_quantiles.
    return_norm : bool, optional
        If True, returns a matplotlib Normalize object for colormapping.
        Default is True.
    
    Returns:
    --------
    tuple: (bin_edges, data_with_quantiles, norm) if return_norm=True
           (bin_edges, data_with_quantiles) if return_norm=False
        - bin_edges (list): The calculated quantile bin edges (length n_quantiles + 1).
        - data_with_quantiles (DataFrame/GeoDataFrame): Copy of input data
          with added 'quantile' column (categorical) and 'quantile_numeric' column (0 to n_quantiles-1).
        - norm (Normalize): Matplotlib Normalizer for consistent colormapping (optional).
    
    Examples:
    ---------
    >>> # Calculate quartiles
    >>> bin_edges, data_with_quartiles, norm = calculate_quantiles(df, 'fatalities', 4)
    
    >>> # Calculate deciles
    >>> bin_edges, data_with_deciles, norm = calculate_quantiles(df, 'events', 10)
    
    >>> # Calculate custom quantiles with custom labels
    >>> bin_edges, data_quintiles = calculate_quantiles(
    ...     df, 'fatalities', 5, 
    ...     labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'],
    ...     return_norm=False
    ... )
    """
    if n_quantiles < 2:
        raise ValueError("n_quantiles must be at least 2")
    
    # Create default labels if not provided
    if labels is None:
        labels = [f'Q{i+1}' for i in range(n_quantiles)]
    elif len(labels) != n_quantiles:
        raise ValueError(f"labels must have length {n_quantiles}, got {len(labels)}")
    
    plot_data = data.copy(deep=True)
    non_nan_data = plot_data[plot_data[measure].notna()]
    
    # Handle cases where there is no data or not enough unique values
    if non_nan_data.empty or non_nan_data[measure].nunique() < n_quantiles:
        min_val = non_nan_data[measure].min() if not non_nan_data.empty else 0
        max_val = non_nan_data[measure].max() if not non_nan_data.empty else 1
        
        if min_val == max_val:  # Avoid division by zero if all values are the same
            min_val -= 0.0001
            max_val += 0.0001
        
        step = (max_val - min_val) / n_quantiles
        bin_edges = [min_val + i * step for i in range(n_quantiles + 1)]
        
        quantile_categories = pd.cut(
            plot_data[measure],
            bins=bin_edges,
            labels=labels,
            include_lowest=True
        )
    else:
        # Calculate percentiles for quantiles
        q_values = [i / n_quantiles for i in range(n_quantiles + 1)]
        percentile_edges = non_nan_data[measure].quantile(q_values).tolist()
        
        # Ensure unique bin edges by adding tiny increments if needed
        bin_edges = []
        for i, edge in enumerate(percentile_edges):
            if i == 0 or edge > bin_edges[-1]:
                bin_edges.append(edge)
            else:
                # Add a tiny increment to make it unique
                bin_edges.append(bin_edges[-1] + 1e-9)
        
        # Use cut to assign data to bins
        quantile_categories = pd.cut(
            plot_data[measure],
            bins=bin_edges,
            labels=labels,
            include_lowest=True
        )
    
    plot_data = plot_data.assign(quantile=quantile_categories.astype(str))
    
    # Create numeric mapping for colormapping (0 to n_quantiles-1)
    quantile_map = {label: i for i, label in enumerate(labels)}
    plot_data['quantile_numeric'] = plot_data['quantile'].map(quantile_map).fillna(-1)
    
    if return_norm:
        norm = Normalize(vmin=0, vmax=n_quantiles - 1)
        return bin_edges, plot_data, norm
    else:
        return bin_edges, plot_data


def calculate_quartiles(data, measure, labels=None, return_norm=True):
    """
    Calculate quartiles (4 bins) for the data.
    
    Quartiles divide data into four equal parts: Q1 (0-25%), Q2 (25-50%), 
    Q3 (50-75%), Q4 (75-100%).
    
    Parameters:
    -----------
    data : pandas.DataFrame or geopandas.GeoDataFrame
        The dataset to calculate quartiles from.
    measure : str
        The numeric column name to base quartiles on.
    labels : list, optional
        Custom labels for the quartile bins. If None, uses ['Q1', 'Q2', 'Q3', 'Q4'].
        Must have length 4.
    return_norm : bool, optional
        If True, returns a matplotlib Normalize object for colormapping.
        Default is True.
    
    Returns:
    --------
    tuple: (bin_edges, data_with_quartiles, norm) if return_norm=True
           (bin_edges, data_with_quartiles) if return_norm=False
        - bin_edges (list): The calculated quartile bin edges (length 5).
        - data_with_quartiles (DataFrame/GeoDataFrame): Copy of input data
          with added 'quantile' and 'quantile_numeric' columns.
        - norm (Normalize): Matplotlib Normalizer for colormapping (optional).
    
    Examples:
    ---------
    >>> bin_edges, df_quartiles, norm = calculate_quartiles(df, 'fatalities')
    >>> bin_edges, df_quartiles = calculate_quartiles(df, 'events', return_norm=False)
    >>> bin_edges, df_quartiles, norm = calculate_quartiles(
    ...     df, 'fatalities', 
    ...     labels=['Low', 'Medium', 'High', 'Very High']
    ... )
    """
    return calculate_quantiles(data, measure, n_quantiles=4, labels=labels, return_norm=return_norm)


def calculate_deciles(data, measure, labels=None, return_norm=True):
    """
    Calculate deciles (10 bins) for the data.
    
    Deciles divide data into ten equal parts: D1 (0-10%), D2 (10-20%), ..., D10 (90-100%).
    
    Parameters:
    -----------
    data : pandas.DataFrame or geopandas.GeoDataFrame
        The dataset to calculate deciles from.
    measure : str
        The numeric column name to base deciles on.
    labels : list, optional
        Custom labels for the decile bins. If None, uses ['D1', 'D2', ..., 'D10'].
        Must have length 10.
    return_norm : bool, optional
        If True, returns a matplotlib Normalize object for colormapping.
        Default is True.
    
    Returns:
    --------
    tuple: (bin_edges, data_with_deciles, norm) if return_norm=True
           (bin_edges, data_with_deciles) if return_norm=False
        - bin_edges (list): The calculated decile bin edges (length 11).
        - data_with_deciles (DataFrame/GeoDataFrame): Copy of input data
          with added 'quantile' and 'quantile_numeric' columns.
        - norm (Normalize): Matplotlib Normalizer for colormapping (optional).
    
    Examples:
    ---------
    >>> bin_edges, df_deciles, norm = calculate_deciles(df, 'fatalities')
    >>> bin_edges, df_deciles = calculate_deciles(df, 'events', return_norm=False)
    """
    if labels is None:
        labels = [f'D{i+1}' for i in range(10)]
    return calculate_quantiles(data, measure, n_quantiles=10, labels=labels, return_norm=return_norm)


def calculate_terciles(data, measure, labels=None, return_norm=True):
    """
    Calculate terciles (3 bins) for the data.
    
    Terciles divide data into three equal parts: T1 (0-33%), T2 (33-67%), T3 (67-100%).
    This function maintains backward compatibility with the existing H3 map visualization.
    
    Parameters:
    -----------
    data : pandas.DataFrame or geopandas.GeoDataFrame
        The dataset to calculate terciles from.
    measure : str
        The numeric column name to base terciles on.
    labels : list, optional
        Custom labels for the tercile bins. If None, uses ['Q1', 'Q2', 'Q3'] 
        (to maintain compatibility with existing code).
        Must have length 3.
    return_norm : bool, optional
        If True, returns a matplotlib Normalize object for colormapping.
        Default is True.
    
    Returns:
    --------
    tuple: (bin_edges, data_with_terciles, norm) if return_norm=True
           (bin_edges, data_with_terciles) if return_norm=False
        - bin_edges (list): The calculated tercile bin edges (length 4).
        - data_with_terciles (DataFrame/GeoDataFrame): Copy of input data
          with added 'quantile' and 'quantile_numeric' columns.
        - norm (Normalize): Matplotlib Normalizer for colormapping (optional).
    
    Examples:
    ---------
    >>> bin_edges, df_terciles, norm = calculate_terciles(df, 'fatalities')
    >>> bin_edges, df_terciles = calculate_terciles(df, 'events', return_norm=False)
    >>> bin_edges, df_terciles, norm = calculate_terciles(
    ...     df, 'fatalities', 
    ...     labels=['Low', 'Medium', 'High']
    ... )
    """
    # Default to 'Q1', 'Q2', 'Q3' for backward compatibility
    if labels is None:
        labels = ['Q1', 'Q2', 'Q3']
    return calculate_quantiles(data, measure, n_quantiles=3, labels=labels, return_norm=return_norm)
