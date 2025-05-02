import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

def create_monte_carlo_chart(simulation_paths: np.ndarray, time_horizon: int, 
                             initial_investment: float, percentiles: List[int] = [5, 25, 50, 75, 95]) -> go.Figure:
    """
    Create a Monte Carlo simulation chart showing multiple paths and percentile lines.
    
    Args:
        simulation_paths: 2D numpy array of simulation paths [n_simulations, time_steps]
        time_horizon: Number of years for the simulation
        initial_investment: Starting portfolio value
        percentiles: List of percentiles to display as lines
    
    Returns:
        Plotly figure object
    """
    # Create a time axis
    years = np.linspace(0, time_horizon, simulation_paths.shape[1])
    
    # Create the figure
    fig = go.Figure()
    
    # Add a sample of individual paths (max 100 for performance)
    sample_size = min(100, simulation_paths.shape[0])
    sample_indices = np.random.choice(simulation_paths.shape[0], sample_size, replace=False)
    
    for idx in sample_indices:
        fig.add_trace(go.Scatter(
            x=years,
            y=simulation_paths[idx, :],
            mode='lines',
            line=dict(width=0.5, color='rgba(100, 100, 100, 0.1)'),
            showlegend=False,
            hoverinfo='none'
        ))
    
    # Add percentile lines
    colors = ['red', 'orange', 'green', 'blue', 'purple']
    
    for i, p in enumerate(percentiles):
        percentile_values = np.percentile(simulation_paths, p, axis=0)
        fig.add_trace(go.Scatter(
            x=years,
            y=percentile_values,
            mode='lines',
            line=dict(width=2, color=colors[i % len(colors)]),
            name=f'{p}th Percentile'
        ))
    
    # Add layout details
    fig.update_layout(
        title='Monte Carlo Simulation: Portfolio Value Projection',
        xaxis_title='Years',
        yaxis_title='Portfolio Value ($)',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_risk_contribution_chart(risk_contributions: Dict[str, float]) -> go.Figure:
    """
    Create a pie chart showing risk contribution by asset.
    
    Args:
        risk_contributions: Dictionary of asset names to risk contribution percentages
    
    Returns:
        Plotly figure object
    """
    # Convert to dataframe
    df = pd.DataFrame({
        'Asset': list(risk_contributions.keys()),
        'Risk Contribution (%)': list(risk_contributions.values())
    })
    
    # Create pie chart
    fig = px.pie(
        df, 
        values='Risk Contribution (%)', 
        names='Asset',
        title='Risk Contribution by Asset',
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    
    return fig

def create_asset_allocation_chart(allocations: Dict[str, float], title: str) -> go.Figure:
    """
    Create a pie chart showing asset allocation.
    
    Args:
        allocations: Dictionary of asset names to allocation percentages
        title: Chart title
    
    Returns:
        Plotly figure object
    """
    # Convert to dataframe
    df = pd.DataFrame({
        'Asset': list(allocations.keys()),
        'Allocation (%)': [val * 100 for val in allocations.values()]  # Convert to percentages
    })
    
    # Create pie chart
    fig = px.pie(
        df, 
        values='Allocation (%)', 
        names='Asset',
        title=title,
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
    )
    
    return fig

def create_scenario_comparison_chart(scenario_results: Optional[Dict[str, Dict[str, float]]] = None) -> go.Figure:
    """
    Create a bar chart comparing different scenarios.
    
    Args:
        scenario_results: Dictionary of scenario names to results dictionaries
    
    Returns:
        Plotly figure object
    """
    # Check if we have valid scenario results
    if not scenario_results:
        # Create placeholder data
        scenario_results = {
            "Base Case": {
                "Expected Return": 0.07,
                "Risk": 0.15,
                "Sharpe Ratio": 0.5
            },
            "Recession": {
                "Expected Return": -0.1,
                "Risk": 0.25,
                "Sharpe Ratio": -0.4
            }
        }
    
    # Extract the data
    scenarios = list(scenario_results.keys())
    if not scenarios:
        # Still no scenarios, create a very simple placeholder
        return px.bar(
            pd.DataFrame({
                'Scenario': ['Base Case', 'Recession'],
                'Metric': ['Expected Return', 'Expected Return'],
                'Value': [0.07, -0.1]
            }),
            x='Scenario',
            y='Value',
            color='Metric',
            title='Scenario Comparison (Placeholder)'
        )
    
    metrics = list(scenario_results[scenarios[0]].keys())
    
    # Create a dataframe for plotting
    data = []
    for scenario in scenarios:
        for metric in metrics:
            data.append({
                'Scenario': scenario,
                'Metric': metric,
                'Value': scenario_results[scenario][metric]
            })
    
    df = pd.DataFrame(data)
    
    # Create the bar chart
    fig = px.bar(
        df,
        x='Scenario',
        y='Value',
        color='Metric',
        barmode='group',
        title='Scenario Comparison',
        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    
    fig.update_layout(
        xaxis_title='Scenario',
        yaxis_title='Value',
        legend_title='Metric',
        hovermode='x'
    )
    
    return fig

def create_historical_price_chart(historical_data: pd.DataFrame, assets: List[str]) -> go.Figure:
    """
    Create a line chart showing historical prices for selected assets.
    
    Args:
        historical_data: DataFrame with DatetimeIndex and columns for each asset
        assets: List of assets to display
    
    Returns:
        Plotly figure object
    """
    # Create the figure
    fig = go.Figure()
    
    # Make a local copy to avoid modifying the original
    data = historical_data.copy()
    
    # Check if we have valid data
    if data.empty:
        # Create an empty chart with a message
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            text="No historical data available",
            showarrow=False,
            font=dict(size=14)
        )
        return fig
    
    # Add each asset as a line
    available_assets = list(data.columns)
    asset_count = 0
    
    for asset in assets:
        if asset in available_assets:
            # Check for missing values
            if data[asset].isna().all():
                continue
                
            try:
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data[asset],
                    mode='lines',
                    name=asset
                ))
                asset_count += 1
            except Exception as e:
                # Silently continue on error
                pass
    
    # If no assets were added, show a message
    if asset_count == 0:
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            text="None of the selected assets found in data",
            showarrow=False,
            font=dict(size=14)
        )
    
    # Update layout
    fig.update_layout(
        title='Historical Asset Prices',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_returns_histogram(returns: pd.DataFrame, asset: str) -> go.Figure:
    """
    Create a histogram of returns for a specific asset.
    
    Args:
        returns: DataFrame with returns data for multiple assets
        asset: Asset to display returns for
    
    Returns:
        Plotly figure object
    """
    # Create figure
    fig = go.Figure()
    
    # Check if we have valid data
    if returns.empty or asset not in returns.columns:
        # Create an empty chart with a message
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            text=f"No returns data available for {asset}",
            showarrow=False,
            font=dict(size=14)
        )
        return fig
    
    # Filter for the specific asset
    asset_returns = returns[asset].dropna()
    
    # Check if we have enough data
    if len(asset_returns) < 5:
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            text=f"Insufficient data for {asset} returns histogram",
            showarrow=False,
            font=dict(size=14)
        )
        return fig
    
    # Create histogram
    fig = px.histogram(
        asset_returns, 
        nbins=min(30, len(asset_returns) // 3),  # Adjust bins based on data size
        title=f'{asset} Returns Distribution',
        labels={'value': 'Return', 'count': 'Frequency'},
        color_discrete_sequence=['#1f77b4']
    )
    
    # Add a normal distribution overlay
    try:
        mean = asset_returns.mean()
        std = asset_returns.std()
        
        if std > 0:  # Avoid division by zero
            x = np.linspace(min(asset_returns), max(asset_returns), 100)
            y = np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
            # Scale to match histogram
            y = y * len(asset_returns) * (max(asset_returns) - min(asset_returns)) / min(30, len(asset_returns) // 3)
            
            fig.add_trace(go.Scatter(
                x=x, 
                y=y,
                mode='lines',
                name='Normal Distribution',
                line=dict(color='red')
            ))
    except Exception as e:
        # Just skip the normal distribution if there's an error
        print(f"Error creating normal distribution overlay: {str(e)}")
    
    # Update layout
    fig.update_layout(
        xaxis_title='Return',
        yaxis_title='Frequency',
        showlegend=True
    )
    
    return fig

def create_optimization_comparison(current_weights: Dict[str, float], 
                                    optimal_weights: Dict[str, float]) -> Tuple[go.Figure, pd.DataFrame]:
    """
    Create a comparison of current and optimal portfolio weights.
    
    Args:
        current_weights: Dictionary of asset names to current weights
        optimal_weights: Dictionary of asset names to optimal weights
    
    Returns:
        Tuple of (chart figure, trades dataframe)
    """
    # Combine the weights into a DataFrame
    df = pd.DataFrame({
        'Asset': list(current_weights.keys()),
        'Current Weight (%)': [w * 100 for w in current_weights.values()],
        'Optimal Weight (%)': [optimal_weights.get(asset, 0) * 100 for asset in current_weights.keys()]
    })
    
    # Calculate the difference
    df['Difference (%)'] = df['Optimal Weight (%)'] - df['Current Weight (%)']
    
    # Create the chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df['Asset'],
        y=df['Current Weight (%)'],
        name='Current Weight',
        marker_color='lightblue'
    ))
    
    fig.add_trace(go.Bar(
        x=df['Asset'],
        y=df['Optimal Weight (%)'],
        name='Optimal Weight',
        marker_color='darkblue'
    ))
    
    # Update layout
    fig.update_layout(
        title='Current vs. Optimal Weights',
        xaxis_title='Asset',
        yaxis_title='Weight (%)',
        barmode='group',
        hovermode='x unified'
    )
    
    # Create the trades DataFrame
    trades_df = df[['Asset', 'Difference (%)']].copy()
    trades_df = trades_df[trades_df['Difference (%)'] != 0]  # Only show assets that need changing
    trades_df = trades_df.sort_values('Difference (%)', ascending=False)
    
    return fig, trades_df

def create_statistics_card(title: str, stats: Dict[str, Any], 
                           formatter: Optional[Dict[str, callable]] = None) -> None:
    """
    Create a card displaying key statistics.
    
    Args:
        title: Card title
        stats: Dictionary of statistic names to values
        formatter: Optional dictionary of stat names to formatter functions
    """
    # Create the card with st.expander
    with st.expander(title, expanded=True):
        # Create two columns
        cols = st.columns(2)
        
        # Distribute stats across columns
        stats_items = list(stats.items())
        half = len(stats_items) // 2 + len(stats_items) % 2
        
        # First column
        for i, (name, value) in enumerate(stats_items[:half]):
            with cols[0]:
                format_func = formatter.get(name, lambda x: x) if formatter else lambda x: x
                st.metric(name, format_func(value))
        
        # Second column
        for i, (name, value) in enumerate(stats_items[half:]):
            with cols[1]:
                format_func = formatter.get(name, lambda x: x) if formatter else lambda x: x
                st.metric(name, format_func(value))