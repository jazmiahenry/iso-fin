import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

def create_portfolio_editor(
    existing_portfolio: Optional[Dict[str, float]] = None,
    key_prefix: str = "",
    validation: bool = True
) -> Dict[str, float]:
    """
    Create a Streamlit portfolio editor component
    
    Args:
        existing_portfolio: Optional dictionary of ticker symbols to weights
        key_prefix: Prefix for Streamlit widget keys
        validation: Whether to validate portfolio weights sum to 1.0
    
    Returns:
        Dictionary of ticker symbols to weights
    """
    st.subheader("Portfolio Composition")
    
    # Initialize portfolio if not provided
    if existing_portfolio is None:
        if 'portfolio' in st.session_state:
            existing_portfolio = st.session_state.portfolio
        else:
            existing_portfolio = {
                "SPY": 0.40,
                "QQQ": 0.20,
                "AGG": 0.30,
                "GLD": 0.10
            }
    
    # Create dataframe from portfolio for editing
    if existing_portfolio is None:
        existing_portfolio = {}
        
    # Handle both dictionary and list formats for portfolio
    if isinstance(existing_portfolio, dict):
        df = pd.DataFrame({
            'Ticker': list(existing_portfolio.keys()),
            'Weight': list(existing_portfolio.values())
        })
    elif isinstance(existing_portfolio, list):
        # Handle list of dictionaries format
        tickers = []
        weights = []
        for asset in existing_portfolio:
            if isinstance(asset, dict):
                name = asset.get('name', '') or asset.get('ticker', '')
                weight = asset.get('weight', 0)
                tickers.append(name)
                weights.append(weight)
        df = pd.DataFrame({
            'Ticker': tickers,
            'Weight': weights
        })
    else:
        # Default empty dataframe
        df = pd.DataFrame({
            'Ticker': [],
            'Weight': []
        })
    
    # Display the current portfolio as a table
    st.dataframe(
        df,
        hide_index=True,
        column_config={
            "Ticker": st.column_config.TextColumn("Ticker Symbol"),
            "Weight": st.column_config.NumberColumn(
                "Weight",
                format="%.2f",
                min_value=0.0,
                max_value=1.0,
                step=0.01
            )
        },
        use_container_width=True
    )
    
    # Add new assets section
    st.subheader("Add New Asset")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # Check if we need to clear the input
        clear_ticker = st.session_state.get(f"{key_prefix}clear_ticker", False)
        initial_value = "" if clear_ticker else st.session_state.get(f"{key_prefix}new_ticker_value", "")
        
        # If there was a clear flag, remove it now
        if clear_ticker:
            st.session_state[f"{key_prefix}clear_ticker"] = False
        
        # Create the input with a value parameter instead of using session_state directly
        new_ticker = st.text_input(
            "Ticker Symbol",
            key=f"{key_prefix}new_ticker",
            value=initial_value,
            placeholder="e.g., AAPL, MSFT, BND"
        )
        
        # Store the current value for next time
        st.session_state[f"{key_prefix}new_ticker_value"] = new_ticker
    
    with col2:
        new_weight = st.number_input(
            "Weight",
            key=f"{key_prefix}new_weight",
            min_value=0.01,
            max_value=1.0,
            value=0.10,
            step=0.01,
            format="%.2f"
        )
    
    with col3:
        st.markdown("##")  # Spacing
        add_button = st.button(
            "Add Asset",
            key=f"{key_prefix}add_asset_button",
            type="primary"
        )
    
    # Process adding new asset
    if add_button and new_ticker:
        # Convert ticker to uppercase
        new_ticker = new_ticker.strip().upper()
        
        # Check if ticker already exists
        if new_ticker in existing_portfolio:
            st.warning(f"Asset {new_ticker} already exists in the portfolio.")
        else:
            # Add the new asset
            existing_portfolio[new_ticker] = new_weight
            
            # Check if we need to normalize
            if validation and sum(existing_portfolio.values()) > 1.0:
                # Normalize weights
                total = sum(existing_portfolio.values())
                for ticker in existing_portfolio:
                    existing_portfolio[ticker] /= total
            
            # Update session state
            st.session_state.portfolio = existing_portfolio
            st.success(f"Added {new_ticker} to the portfolio.")
            
            # Initialize a flag in session state to clear the input on next run
            # This avoids modifying a widget key that's already been used
            st.session_state[f"{key_prefix}clear_ticker"] = True
            
            # Rerun to update the UI
            st.rerun()
    
    # Remove assets section
    st.subheader("Remove Asset")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        remove_ticker = st.selectbox(
            "Select Asset to Remove",
            options=list(existing_portfolio.keys()),
            key=f"{key_prefix}remove_ticker"
        )
    
    with col2:
        st.markdown("##")  # Spacing
        remove_button = st.button(
            "Remove Asset",
            key=f"{key_prefix}remove_asset_button",
            type="secondary"
        )
    
    # Process removing asset
    if remove_button and remove_ticker and len(existing_portfolio) > 1:
        # Remove the asset
        del existing_portfolio[remove_ticker]
        
        # Check if we need to normalize
        if validation:
            # Normalize weights to sum to 1
            total = sum(existing_portfolio.values())
            for ticker in existing_portfolio:
                existing_portfolio[ticker] /= total
        
        # Update session state
        st.session_state.portfolio = existing_portfolio
        st.success(f"Removed {remove_ticker} from the portfolio.")
                
        # Rerun to update the UI - but don't try to modify the selectbox value
        st.rerun()
    elif remove_button and len(existing_portfolio) <= 1:
        st.error("Cannot remove the last asset from the portfolio.")
    
    # Adjust weights section
    st.subheader("Adjust Weights")
    
    # Create a form for adjusting all weights at once
    with st.form(key=f"{key_prefix}adjust_weights_form"):
        # Create a dictionary to store the weight inputs
        weight_inputs = {}
        
        # Create a column for each asset
        columns = st.columns(min(4, len(existing_portfolio)))
        
        # Distribute assets across columns
        for i, (ticker, weight) in enumerate(existing_portfolio.items()):
            col_index = i % len(columns)
            with columns[col_index]:
                weight_inputs[ticker] = st.number_input(
                    ticker,
                    min_value=0.0,
                    max_value=1.0,
                    value=float(weight),
                    step=0.01,
                    format="%.2f",
                    key=f"{key_prefix}weight_{ticker}"
                )
        
        # Add a submit button
        submit_button = st.form_submit_button(
            "Update Weights",
            type="primary"
        )
        
        # Process form submission
        if submit_button:
            # Update the portfolio with new weights
            for ticker, weight in weight_inputs.items():
                existing_portfolio[ticker] = weight
            
            # Check if we need to normalize
            if validation:
                # Calculate the sum of weights
                total_weight = sum(existing_portfolio.values())
                
                # If the sum is not 1.0 (allowing for floating point errors)
                if abs(total_weight - 1.0) > 0.0001:
                    # Normalize weights
                    for ticker in existing_portfolio:
                        existing_portfolio[ticker] /= total_weight
                    
                    st.info(f"Weights have been normalized to sum to 1.0 (original sum was {total_weight:.2f}).")
            
            # Update session state
            st.session_state.portfolio = existing_portfolio
            st.success("Portfolio weights updated successfully.")
    
    # Reset portfolio button
    reset_btn = st.button("Reset Portfolio", key=f"{key_prefix}reset_portfolio")
    
    # Store if reset was clicked
    if reset_btn:
        st.session_state[f"{key_prefix}reset_clicked"] = True
    
    # Show confirmation if reset was clicked
    if st.session_state.get(f"{key_prefix}reset_clicked", False):
        confirm = st.checkbox("Confirm reset - this will revert to the default portfolio", key=f"{key_prefix}confirm_reset")
        
        if confirm:
            # Reset to default portfolio
            default_portfolio = {
                "SPY": 0.40,
                "QQQ": 0.20,
                "AGG": 0.30,
                "GLD": 0.10
            }
            
            # Update session state
            st.session_state.portfolio = default_portfolio
            st.success("Portfolio reset to default.")
            
            # Update the return value
            existing_portfolio = default_portfolio
            
            # Clear reset state
            st.session_state[f"{key_prefix}reset_clicked"] = False
            
            # Rerun to update the UI
            st.rerun()
        
        # Add a cancel button
        cancel = st.button("Cancel", key=f"{key_prefix}cancel_reset")
        if cancel:
            st.session_state[f"{key_prefix}reset_clicked"] = False
            st.rerun()
    
    # Portfolio summary
    st.subheader("Portfolio Summary")
    
    # Calculate total
    total_weight = sum(existing_portfolio.values())
    
    # Format summary message based on total
    if abs(total_weight - 1.0) <= 0.0001:
        st.success(f"Total weight: {total_weight:.2f} (Correctly sums to 1.0)")
    else:
        st.warning(f"Total weight: {total_weight:.2f} (Should sum to 1.0)")
    
    # Display as a pie chart
    if isinstance(existing_portfolio, dict):
        # Create pie chart dataframe from dictionary
        pie_df = pd.DataFrame({
            'Asset': list(existing_portfolio.keys()),
            'Weight': list(existing_portfolio.values())
        })
    else:
        # Just use the original dataframe but rename columns for the pie chart
        pie_df = df.copy()
        pie_df.columns = ['Asset', 'Weight']
    
    # Sort by weight
    pie_df = pie_df.sort_values('Weight', ascending=False)
    
    # Display the pie chart if we have data
    if not pie_df.empty:
        # Use plotly express to create a pie chart
        import plotly.express as px
        fig = px.pie(
            pie_df, 
            values='Weight', 
            names='Asset',
            title='Portfolio Allocation',
            hole=0.3  # Donut chart style
        )
        # Set percentage display format
        fig.update_traces(textposition='inside', textinfo='percent+label')
        # Display the chart
        st.plotly_chart(fig, use_container_width=True)
    
    # Convert the portfolio to a dictionary format for consistency
    portfolio_dict = {}
    
    # If the input was a list, convert the output to dictionary format
    if isinstance(existing_portfolio, list):
        for asset in existing_portfolio:
            if isinstance(asset, dict):
                name = asset.get('name', '') or asset.get('ticker', '')
                if name:
                    portfolio_dict[name] = asset.get('weight', 0)
    else:
        portfolio_dict = existing_portfolio
    
    # Return the updated portfolio as a dictionary
    return portfolio_dict