import streamlit as st
from iso_financial_mvp.streamlit.utils.state_manager import logout, clear_simulation_results

def create_sidebar():
    """Create and configure the sidebar for navigation and controls"""
    with st.sidebar:
        st.title("ISO Financial")
        
        # Only show navigation when authenticated
        if st.session_state.authenticated:
            # Active page indicator
            active_page = st.radio(
                "Navigate To:",
                ["Home", "Dashboard", "Reports", "Financial Educator", "About"],
                index=0,
                key="navigation"
            )
            
            # Keep track of current tier in session state for persistence
            user_tier = st.session_state.user_tier
            
            # Show user tier in the sidebar
            st.subheader("Account Status")
            if user_tier == "premium":
                st.success("Premium Access")
            else:
                st.info("Free Account")
            
            # Navigation links - using standard paths
            if active_page == "Dashboard":
                st.page_link("iso_financial_mvp/streamlit/pages/dashboard.py", label="Go to Dashboard", icon="📊")
            elif active_page == "Reports":
                st.page_link("iso_financial_mvp/streamlit/pages/reports.py", label="Go to Reports", icon="📄")
            elif active_page == "Financial Educator":
                st.page_link("iso_financial_mvp/streamlit/pages/financial_educator.py", label="Financial Educator", icon="🧠")
            elif active_page == "About": 
                st.page_link("iso_financial_mvp/streamlit/pages/settings.py", label="About", icon="ℹ️")
            # Home is the default page (app.py)
            elif active_page == "Home":
                st.page_link("app.py", label="Return to Home", icon="🏠")
            
            # Logout button
            if st.button("Logout"):
                logout()
                st.rerun()
            
            # Display disclaimer about academic/demo nature if needed
            st.markdown("---")
            st.caption("This app is for educational purposes only. Not financial advice.")
        
        # Footer logo and version info
        st.sidebar.markdown("---")
        st.sidebar.image("https://via.placeholder.com/150x50?text=ISO+Financial", width=150)
        st.sidebar.caption("Version 1.0.0")