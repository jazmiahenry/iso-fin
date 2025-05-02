import streamlit as st
from datetime import datetime

def initialize_session_state():
    """Initialize session state variables if they don't exist"""
    
    # Authentication state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    # User tier (free or premium)
    # Note: If user_tier is already set, we don't overwrite it
    # This helps maintain tier persistence across page navigations
    if 'user_tier' not in st.session_state:
        # Default to free if not set
        st.session_state.user_tier = "free"
    
    # Usage tracking for free tier limitations
    if 'simulations_count' not in st.session_state:
        st.session_state.simulations_count = 0
    
    if 'monthly_queries' not in st.session_state:
        st.session_state.monthly_queries = 0
        
    if 'query_reset_date' not in st.session_state:
        st.session_state.query_reset_date = datetime.now().strftime("%Y-%m")
        
    # Financial Educator usage tracking
    if 'gpt_queries_count' not in st.session_state:
        st.session_state.gpt_queries_count = 0
        
    if 'education_history' not in st.session_state:
        st.session_state.education_history = []
    
    # Portfolio data - initialize with default values
    if 'portfolio' not in st.session_state or st.session_state.portfolio is None:
        st.session_state.portfolio = {
            "SPY": 0.40,  # S&P 500 ETF
            "QQQ": 0.20,  # Nasdaq ETF
            "AGG": 0.30,  # US Bond ETF
            "GLD": 0.10   # Gold ETF
        }
    
    # Initial investment
    if 'initial_investment' not in st.session_state:
        st.session_state.initial_investment = 100000
    
    # Portfolio editor visibility
    if 'show_portfolio_editor' not in st.session_state:
        st.session_state.show_portfolio_editor = False
    
    # Simulation flag
    if 'simulations_run' not in st.session_state:
        st.session_state.simulations_run = False
    
    # Individual simulation results
    if 'monte_carlo_results' not in st.session_state:
        st.session_state.monte_carlo_results = None
        
    if 'scenario_results' not in st.session_state:
        st.session_state.scenario_results = None
        
    if 'risk_results' not in st.session_state:
        st.session_state.risk_results = None
        
    if 'optimization_results' not in st.session_state:
        st.session_state.optimization_results = None
    
    # Last query for reference
    if 'last_query' not in st.session_state:
        st.session_state.last_query = ""
    
    # Report generation state
    if 'report_path' not in st.session_state:
        st.session_state.report_path = None
        
    # User profile information
    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = {}
        
    # App settings
    if 'app_settings' not in st.session_state:
        st.session_state.app_settings = {
            'theme': 'Light',
            'chart_color': 'Default',
            'table_display': 'Compact',
            'date_format': 'MM/DD/YYYY'
        }
        
    # Simulation settings
    if 'sim_settings' not in st.session_state:
        st.session_state.sim_settings = {
            'default_sim_count': 1000,
            'default_time_horizon': 10,
            'risk_model': 'Standard',
            'correlation_model': 'Historical'
        }
        
    # Notification settings - tier specific
    if 'notif_settings' not in st.session_state:
        # Default settings based on tier
        if st.session_state.user_tier == "premium":
            st.session_state.notif_settings = {
                'receive_reports': True,
                'market_alerts': True,
                'portfolio_alerts': True,
                'newsletter': True,
                'market_movement': 3.0,
                'portfolio_movement': 2.0,
                'frequency': 'Daily'
            }
        else:
            st.session_state.notif_settings = {
                'receive_reports': True,
                'market_alerts': False,
                'portfolio_alerts': True,
                'newsletter': False,
                'market_movement': 5.0,
                'portfolio_movement': 3.0,
                'frequency': 'Monthly'
            }

def clear_simulation_results():
    """Clear simulation results from session state"""
    # Reset simulation flag
    st.session_state.simulations_run = False
    
    # Clear individual simulation results
    st.session_state.monte_carlo_results = None
    st.session_state.scenario_results = None
    st.session_state.risk_results = None
    st.session_state.optimization_results = None
    
    # Clear last query
    st.session_state.last_query = ""

def clear_portfolio():
    """Clear portfolio data from session state"""
    # Reset to default portfolio
    st.session_state.portfolio = {
        "SPY": 0.40,  # S&P 500 ETF
        "QQQ": 0.20,  # Nasdaq ETF
        "AGG": 0.30,  # US Bond ETF
        "GLD": 0.10   # Gold ETF
    }
    
    # Reset initial investment
    st.session_state.initial_investment = 100000
    
    # Also clear simulation results as they depend on portfolio
    clear_simulation_results()

def logout():
    """Reset all session state for logout"""
    # Store the tier temporarily if we want to preserve it
    current_tier = st.session_state.get("user_tier", "free")
    
    # Clear all session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    # Reinitialize session state
    initialize_session_state()
    
    # Set authentication to false
    st.session_state.authenticated = False

def get_tier_limits():
    """Returns the limits based on user tier"""
    free_tier_limits = {
        "max_assets": 5,
        "max_time_horizon": 5,
        "simulation_count": 100,
        "max_simulations_per_session": 1,
        "max_queries_per_month": 1,
        "max_gpt_queries_per_session": 0,  # No Financial Educator queries for free tier
        "report_types": ["free"],
        "available_scenarios": ["recession"]
    }
    
    premium_tier_limits = {
        "max_assets": float('inf'),  # Unlimited
        "max_time_horizon": 30,
        "simulation_count": None,  # User can choose
        "max_simulations_per_session": float('inf'),  # Unlimited
        "max_queries_per_month": float('inf'),  # Unlimited
        "max_gpt_queries_per_session": 10,  # 10 Financial Educator queries per day for premium
        "report_types": ["free", "premium"],
        "available_scenarios": ["recession", "stagflation", "marketCrash", "highGrowth", "custom"]
    }
    
    user_tier = st.session_state.get("user_tier", "free")
    return premium_tier_limits if user_tier == "premium" else free_tier_limits

def enforce_tier_limits():
    """Enforce all tier limits consistently"""
    user_tier = st.session_state.get("user_tier", "free")
    limits = get_tier_limits()
    
    # Check monthly query limit reset date
    current_month = datetime.now().strftime("%Y-%m")
    if st.session_state.get("query_reset_date") != current_month:
        st.session_state.query_reset_date = current_month
        st.session_state.monthly_queries = 0
    
    # Reset simulations count for a new session if it doesn't exist
    if 'simulations_count' not in st.session_state:
        st.session_state.simulations_count = 0
    
    # Enforce portfolio size limit
    if user_tier == "free" and 'portfolio' in st.session_state and len(st.session_state.portfolio) > limits["max_assets"]:
        # Keep only the assets with highest weights
        sorted_assets = sorted(st.session_state.portfolio.items(), key=lambda x: x[1], reverse=True)
        limited_portfolio = {k: v for k, v in sorted_assets[:limits["max_assets"]]}
        
        # Normalize weights to sum to 1
        total_weight = sum(limited_portfolio.values())
        if total_weight > 0:
            limited_portfolio = {k: v/total_weight for k, v in limited_portfolio.items()}
        
        st.session_state.portfolio = limited_portfolio