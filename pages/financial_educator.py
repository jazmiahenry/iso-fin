"""
Financial Educator page using a finetuned GPT-4 model for financial education.
"""

import streamlit as st

# Set page config - must be the first Streamlit command
st.set_page_config(
    page_title="ISO Financial Educator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

import sys
import os
from typing import Dict, Any

# Add necessary paths to ensure all modules are accessible
# The parent directory of the pages directory is the project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Print path debugging info (can be removed later)
print(f"Financial Educator page path: {__file__}")
print(f"Project root: {project_root}")
print(f"Current working directory: {os.getcwd()}")

# Ensure Python can find our modules
os.environ["PYTHONPATH"] = f"{project_root}:{os.environ.get('PYTHONPATH', '')}"

# Import both agent types with robust error handling
# Initialize agent variables so they exist regardless of import success
BasicGPTAgent = None
FinancialGPTAgent = None

try:
    # Try absolute imports first
    from iso_financial_mvp.gpt_agent.basic_gpt import BasicGPTAgent
    from iso_financial_mvp.gpt_agent.financial_agent import FinancialGPTAgent
    st.success("Successfully imported agent modules")
except ImportError:
    try:
        # Try direct imports from project root structure
        import importlib.util
        
        # Find the gpt_agent directory (using the project root)
        gpt_dir = os.path.join(project_root, "iso_financial_mvp", "gpt_agent")
        
        # Check if directory exists
        if not os.path.exists(gpt_dir):
            st.warning(f"gpt_agent directory not found at {gpt_dir}")
            raise ImportError("Could not find gpt_agent directory")
        
        # Import basic_gpt.py
        basic_path = os.path.join(gpt_dir, "basic_gpt.py")
        if os.path.exists(basic_path):
            spec = importlib.util.spec_from_file_location("basic_gpt", basic_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            BasicGPTAgent = module.BasicGPTAgent
            st.success(f"Successfully imported BasicGPTAgent from {basic_path}")
        else:
            st.warning(f"basic_gpt.py not found at {basic_path}")
            
        # Import financial_agent.py
        fin_path = os.path.join(gpt_dir, "financial_agent.py") 
        if os.path.exists(fin_path):
            spec = importlib.util.spec_from_file_location("financial_agent", fin_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            FinancialGPTAgent = module.FinancialGPTAgent
            st.success(f"Successfully imported FinancialGPTAgent from {fin_path}")
        else:
            st.warning(f"financial_agent.py not found at {fin_path}")
    except Exception as e:
        st.error(f"Failed to import agent modules. Error: {str(e)}")
        # The variables are already initialized as None

# Import state management
from iso_financial_mvp.streamlit.utils.state_manager import initialize_session_state, get_tier_limits
from iso_financial_mvp.streamlit.components.sidebar import create_sidebar


# Initialize session state
initialize_session_state()

# Create sidebar
create_sidebar()

# Page title
st.title("🧠 Financial Educator")
st.markdown("""
This educational tool uses a sophisticated AI model finetuned on financial data to provide 
in-depth financial education and analysis. Ask any financial question and get 
detailed insights with calculations as needed.
""")

# Check if API key is available
api_key_available = os.environ.get("OPENAI_API_KEY") is not None
if not api_key_available:
    st.error("⚠ OpenAI API key not found! Please set the OPENAI_API_KEY environment variable.")

# Initialize Financial Educator - try finetuned model first, then fall back to basic model
advisor = None
if api_key_available:
    # First try to initialize with finetuned model
    try:
        advisor = FinancialGPTAgent()
        st.success("Financial Educator loaded successfully with finetuned model!")
    except Exception as e:
        st.warning(f"Could not initialize Financial Educator with finetuned model: {str(e)}")
        
        # Fall back to basic model
        try:
            advisor = BasicGPTAgent()
            st.success("Financial Educator loaded successfully with standard GPT-4.1 model!")
        except Exception as e:
            st.error(f"Could not initialize Financial Educator: {str(e)}")

# Check tier access
limits = get_tier_limits()
has_premium_access = st.session_state.user_tier == "premium"

# Enforce premium tier restriction
if not has_premium_access and limits.get("max_gpt_queries_per_session", 0) <= 0:
    st.error("Financial Educator is a premium feature. Please upgrade to access.")
    st.stop()

# Show query limit warning for free tier
if not has_premium_access:
    query_limit = limits.get("max_gpt_queries_per_session", 1)
    queries_used = st.session_state.get("gpt_queries_count", 0)
    st.warning(f"Free tier: {queries_used}/{query_limit} queries used")
    
    # Stop if limit reached
    if queries_used >= query_limit:
        st.error("You have used all your free queries. Please upgrade for unlimited access.")
        st.stop()

# Main query interface
with st.container():
    # User input
    user_query = st.text_area(
        "Enter your financial question:",
        placeholder="Example: How should I adjust my portfolio allocation during high inflation?",
        height=100
    )
    
    # Optional context
    with st.expander("Add context information (optional)"):
        # Include portfolio context
        if 'portfolio' in st.session_state and st.session_state.portfolio:
            st.checkbox("Include my current portfolio", key="include_portfolio", value=True)
        
        # Market conditions
        market_condition = st.selectbox(
            "Current market condition",
            ["Not specified", "Bull market", "Bear market", "High volatility", "Low volatility", "Recession", "Recovery"]
        )
        
        # Risk tolerance
        risk_tolerance = st.selectbox(
            "Your risk tolerance",
            ["Not specified", "Conservative", "Moderate", "Aggressive"]
        )
        
        # Investment horizon
        investment_horizon = st.selectbox(
            "Your investment horizon",
            ["Not specified", "Short-term (< 1 year)", "Medium-term (1-5 years)", "Long-term (5+ years)"]
        )
    
    # Submit button
    if st.button("Get Financial Education", type="primary"):
        if not user_query:
            st.error("Please enter a question.")
        elif not advisor:
            st.error("Financial Educator is not available. Please check API keys and agent initialization.")
        else:
            # Show processing spinner
            with st.spinner("Consulting the Financial Educator..."):
                # Prepare context
                context = {}
                
                # Add portfolio if selected
                if st.session_state.get("include_portfolio", False) and 'portfolio' in st.session_state:
                    context["portfolio"] = st.session_state.portfolio
                
                # Add other context
                if market_condition != "Not specified":
                    context["market_condition"] = market_condition
                
                if risk_tolerance != "Not specified":
                    context["risk_tolerance"] = risk_tolerance
                
                if investment_horizon != "Not specified":
                    context["investment_horizon"] = investment_horizon
                
                # Increment query count for free tier
                if not has_premium_access:
                    st.session_state["gpt_queries_count"] = st.session_state.get("gpt_queries_count", 0) + 1
                
                # Get response from model
                result = advisor.answer_question(user_query, context)
                
                # Display the response
                st.markdown("### Response")
                st.markdown(result["response"])
                
                # Show calculations if any were performed
                calculations = result.get("calculations", [])
                if calculations:
                    st.markdown("### Calculations")
                    for i, calc in enumerate(calculations):
                        with st.expander(f"Calculation {i+1}: {calc.get('tool', 'Financial calculation')}"):
                            st.json(calc)
                
                # Save to session history
                if 'education_history' not in st.session_state:
                    st.session_state.education_history = []
                
                st.session_state.education_history.append({
                    "query": user_query,
                    "context": context,
                    "response": result
                })

# Show history
if st.session_state.get('education_history', []):
    with st.expander("Previous Questions", expanded=False):
        for i, item in enumerate(reversed(st.session_state.education_history)):
            st.markdown(f"#### Question {len(st.session_state.education_history) - i}")
            st.markdown(f"**Q:** {item['query']}")
            st.markdown(f"**A:** {item['response']['response']}")
            st.markdown("---")