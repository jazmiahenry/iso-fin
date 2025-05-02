import streamlit as st
import sys
import os

# Add parent directory to path to import from other modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Print path debugging info (can be removed later)
print(f"Settings page path: {__file__}")
print(f"Project root: {project_root}")
print(f"Current working directory: {os.getcwd()}")

# Ensure Python can find our modules
os.environ["PYTHONPATH"] = f"{project_root}:{os.environ.get('PYTHONPATH', '')}"

# Import state management
from iso_financial_mvp.streamlit.utils.state_manager import initialize_session_state

# Set page config
st.set_page_config(
    page_title="About ISO Financial",
    page_icon="ℹ️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state if not already done
initialize_session_state()

# Set user tier based on query parameter only if explicitly provided
tier = st.query_params.get("tier", None)
if tier in ["free", "premium"]:
    st.session_state.user_tier = tier

# Check if user is authenticated
if not st.session_state.get('authenticated', False):
    st.warning("Please log in to access this page.")
    st.stop()

def about_main():
    """Main about page function"""
    st.title("ℹ️ About ISO Financial")
    
    st.write("""
    **ISO Financial** provides cutting-edge portfolio analysis tools using advanced simulation techniques including
    Monte Carlo simulation, scenario analysis, risk parity, Conditional Value at Risk (CVaR), and Bayesian scenario ranking.
    
    Our tools help investors understand the range of possible outcomes for their portfolios and make more informed decisions
    about asset allocation, risk management, and investment strategy.
    """)
    
    # Version information
    st.subheader("Version Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Application Version", "1.0.0")
        st.metric("API Version", "v1")
    
    with col2:
        st.metric("Last Updated", "April 2025")
        st.metric("Database Version", "2.1.3")
    
    # Contact information
    st.subheader("Contact Us")
    
    st.write("""
    **Email:** hello@isoai.co  
    
    For customer support, please include your account email in all correspondence.
    """)

# Run the about page
about_main()