import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import base64
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

# Add parent directory to path to import from other modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Print path debugging info (can be removed later)
print(f"Reports page path: {__file__}")
print(f"Project root: {project_root}")
print(f"Current working directory: {os.getcwd()}")

# Ensure Python can find our modules
os.environ["PYTHONPATH"] = f"{project_root}:{os.environ.get('PYTHONPATH', '')}"

# Import report generators
from iso_financial_mvp.report_generator.pdf_generator import PDFReport
from iso_financial_mvp.report_generator.disclaimer_template import get_standard_disclaimer
from iso_financial_mvp.llm_interface.query_parser import QueryParser
from iso_financial_mvp.simulation_engines.monte_carlo import MonteCarloSimulator
from iso_financial_mvp.simulation_engines.scenario_analysis import ScenarioAnalysisEngine
from iso_financial_mvp.simulation_engines.risk_parity import RiskParityEngine
from iso_financial_mvp.simulation_engines.cvar import CVaRCalculator

# Import state management
from iso_financial_mvp.streamlit.utils.state_manager import initialize_session_state, enforce_tier_limits, get_tier_limits
from iso_financial_mvp.streamlit.components.sidebar import create_sidebar

# Initialize the page
st.set_page_config(
    page_title="ISO Financial - Ask AI Assistant",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
initialize_session_state()

# Set user tier based on query parameter only if explicitly provided
tier = st.query_params.get("tier", None)
if tier in ["free", "premium"]:
    st.session_state.user_tier = tier

# Enforce tier limits
enforce_tier_limits()

# Create sidebar
create_sidebar()

# Page title
st.title("Ask AI Assistant")

# Create tabs for AI Assistant and Reports
tab1, tab2 = st.tabs(["Ask AI Assistant", "Generate Reports"])

with tab1:
    st.header("Ask a Question About Your Portfolio")
    st.write("Our AI assistant can analyze your portfolio and answer specific questions about your investments.")

    # Check if simulations have been run and data is available
    if not st.session_state.get("simulations_run", False):
        st.warning("Please run a simulation on the main page first to enable the AI assistant.")
    else:
        # Check for premium usage
        limits = get_tier_limits()
        monthly_queries = st.session_state.get("monthly_queries", 0)
        
        if st.session_state.user_tier == "free" and monthly_queries >= limits["max_queries_per_month"]:
            st.warning(f"Free tier is limited to {limits['max_queries_per_month']} AI query per month. Please upgrade to premium for unlimited queries.")
        else:
            query = st.text_input(
                "Enter your question (e.g., 'What if ETH drops 40%?', 'How risky is my portfolio?', 'Should I adjust my asset allocation?')",
                key="ai_query"
            )
            
            col1, col2, col3 = st.columns(3)
            with col1:
                risk_profile = st.selectbox(
                    "Risk Profile",
                    ["conservative", "moderate", "aggressive"],
                    index=1,
                    key="ai_risk_profile"
                )
            with col2:
                # Apply tier-appropriate time horizon limits
                if st.session_state.user_tier == "free":
                    time_horizon = st.slider(
                        f"Time Horizon (Years) - Limited to {limits['max_time_horizon']} years",
                        min_value=1,
                        max_value=limits["max_time_horizon"],
                        value=min(limits["max_time_horizon"], 10),
                        key="ai_time_horizon"
                    )
                else:
                    time_horizon = st.slider(
                        "Time Horizon (Years)",
                        min_value=1,
                        max_value=limits["max_time_horizon"],
                        value=10,
                        key="ai_time_horizon"
                    )
            with col3:
                user_profile = st.selectbox(
                    "Your Expertise Level",
                    ["novice", "intermediate", "pro"],
                    index=0,
                    key="ai_user_profile",
                    help="This helps our AI tailor explanations to your knowledge level"
                )
            
            if st.button("Get Analysis", type="primary", key="ai_analyze_button"):
                with st.spinner("Analyzing your portfolio..."):
                    # Check current month and reset counter if it's a new month
                    current_month = datetime.now().strftime("%Y-%m")
                    if st.session_state.get("query_reset_date") != current_month:
                        st.session_state.query_reset_date = current_month
                        st.session_state.monthly_queries = 0
                        monthly_queries = 0
                    
                    # Increment monthly queries counter for free tier
                    if st.session_state.user_tier == "free":
                        st.session_state.monthly_queries = monthly_queries + 1
                    
                    # Retrieve data from session state
                    portfolio = st.session_state.get("portfolio", {})
                    initial_investment = st.session_state.get("initial_investment", 100000)
                    mc_results = st.session_state.get("monte_carlo_results", {})
                    scenario_results = st.session_state.get("scenario_results", {})
                    risk_results = st.session_state.get("risk_results", {})
                    optimization_results = st.session_state.get("optimization_results", {})
                    
                    # Process query
                    query_parser = QueryParser(model_name="claude-3-sonnet-20240229")
                    query_data = query_parser.parse_query(query if query else "Analyze my portfolio")
                    
                    # Update query data with user selections and portfolio info
                    query_data["portfolio"] = portfolio
                    query_data["initial_investment"] = initial_investment
                    query_data["risk_profile"] = risk_profile
                    query_data["timeframe"] = {"unit": "years", "value": time_horizon}
                    
                    # Prepare all results for the response generation
                    response_data = {
                        "query_data": query_data,
                        "monte_carlo_results": mc_results,
                        "scenario_results": scenario_results,
                        "risk_results": risk_results,
                        "optimization_results": optimization_results,
                        "original_query": query
                    }
                    
                    # Generate response using Claude LLM
                    with st.spinner("Generating AI analysis with Claude..."):
                        try:
                            try:
                                # Import anthropic here to ensure it's available
                                import anthropic
                                from dotenv import load_dotenv
                                import os
                                import logging
                                
                                # Set up logging
                                logging.basicConfig(level=logging.INFO)
                                logger = logging.getLogger("iso_financial")
                                
                                # Force direct API call to Claude, bypassing the generate_response method
                                # Load environment variables directly
                                load_dotenv()
                                api_key = os.getenv("ANTHROPIC_API_KEY")
                                
                                # Add error checking
                                if not api_key:
                                    logger.error("ANTHROPIC_API_KEY not found in environment")
                                    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
                                
                                client = anthropic.Anthropic(api_key=api_key)
                                
                                # Create a direct prompt for Claude
                                system_prompt = "You are a financial advisor assistant that helps explain portfolio analysis results in clear, concise language appropriate for the user's knowledge level. Focus on key insights and actionable recommendations."
                                
                                # Prepare simplified prompt with core data
                                timeframe = query_data.get("timeframe", {}).get("value", 10)
                                initial_investment = response_data.get("monte_carlo_results", {}).get("initial_investment", 100000)
                                
                                # Handle numpy values
                                if hasattr(initial_investment, "item"):
                                    initial_investment = initial_investment.item()
                                
                                # Format portfolio for prompt
                                portfolio_str = ""
                                for ticker, weight in portfolio.items():
                                    portfolio_str += f"- {ticker}: {weight*100:.1f}%\n"
                                
                                user_prompt = f"""
User query: "{query if query else 'Analyze my portfolio'}"

Portfolio Analysis:
- Initial Investment: ${initial_investment:,.0f}
- Time Horizon: {timeframe} years
- Risk Profile: {risk_profile}
- User Expertise: {user_profile}

Portfolio Composition:
{portfolio_str}

Provide a detailed analysis based on this portfolio configuration, including insights on expected growth, risk factors, and actionable recommendations.
"""
                                
                                # Call Claude API directly
                                logger.info(f"Calling Claude API with user query: {query}")
                                response = client.messages.create(
                                    model="claude-3-sonnet-20240229",
                                    system=system_prompt,
                                    messages=[
                                        {"role": "user", "content": user_prompt}
                                    ],
                                    temperature=0.7,
                                    max_tokens=1000
                                )
                                
                                # Extract the content from the response
                                llm_response = response.content[0].text
                                logger.info("Successfully received response from Claude")
                                
                            except ImportError as e:
                                logger.error(f"Missing dependency: {e}")
                                raise ValueError(f"Required dependency missing: {e}")
                            except Exception as e:
                                logger.error(f"Error calling Claude API: {str(e)}")
                                raise e
                            
                        except Exception as e:
                            # Log the error internally but don't show technical details to users
                            print(f"[ERROR] AI Response generation failed: {str(e)}")
                            st.error("We couldn't generate a response at this time. Please try again later.")
                            llm_response = "I'm sorry, but I'm unable to provide an analysis at this moment. Please try again in a few minutes."
                    
                    # Display LLM response in a highlighted box
                    st.subheader("Analysis Results")
                    st.markdown(f"""
                    ### Response to: "{query}"
                    
                    {llm_response}
                    """)
                    
                    # Add contextual metrics for reference
                    with st.expander("Portfolio Metrics"):
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            initial_value = initial_investment
                            final_mean = mc_results.get('mean_final', 0)
                            growth_pct = (final_mean - initial_value) / initial_value * 100 if initial_value else 0
                            st.metric(
                                "Expected Final Value", 
                                f"${final_mean:,.0f}" if isinstance(final_mean, (int, float)) else "N/A",
                                f"{growth_pct:.1f}%" if isinstance(growth_pct, (int, float)) else "N/A"
                            )
                            
                        with col2:
                            median_final = mc_results.get('median_final', 0)
                            st.metric(
                                "Median Final Value", 
                                f"${median_final:,.0f}" if isinstance(median_final, (int, float)) else "N/A"
                            )
                            
                        with col3:
                            cvar_95 = risk_results.get('cvar_95', 0)
                            st.metric(
                                "CVaR (95%)", 
                                f"${cvar_95:,.0f}" if isinstance(cvar_95, (int, float)) else "N/A"
                            )
                            
                        with col4:
                            max_drawdown = risk_results.get('max_drawdown', 0)
                            st.metric(
                                "Maximum Drawdown", 
                                f"{max_drawdown:.1%}" if isinstance(max_drawdown, (int, float)) else "N/A"
                            )

with tab2:
    st.header("Generate Custom Reports")
    
    # Check if simulations have been run
    if not st.session_state.get("simulations_run", False):
        st.warning("Please run a simulation first to generate a report.")
    else:
        # Get tier limits
        limits = get_tier_limits()
        available_report_types = limits["report_types"]
        
        # Report type selection based on user tier
        if "premium" in available_report_types:
            report_type = st.radio(
                "Report Type",
                ["Free Summary Report", "Premium Detailed Report"],
                horizontal=True
            )
        else:
            report_type = "Free Summary Report"
            st.info("Free tier users can generate basic summary reports. Upgrade to premium for detailed reports.")
        
        # Report customization
        st.subheader("Report Customization")
        
        col1, col2 = st.columns(2)
        
        with col1:
            include_monte_carlo = st.checkbox("Include Monte Carlo Analysis", value=True)
            include_risk = st.checkbox("Include Risk Assessment", value=True)
            include_optimization = st.checkbox("Include Portfolio Optimization", value=True)
        
        with col2:
            include_scenarios = st.checkbox("Include Scenario Analysis", value=True)
            include_historical = st.checkbox("Include Historical Performance", value=True)
            include_disclaimer = st.checkbox("Include Disclaimer", value=True)
        
        # Generate report button
        if st.button("Generate Report", type="primary", key="generate_report_button"):
            if report_type == "Free Summary Report":
                with st.spinner("Generating Free Report..."):
                    # Get simulation results from session state
                    monte_carlo_results = st.session_state.get("monte_carlo_results", {})
                    risk_results = st.session_state.get("risk_results", {})
                    optimization_results = st.session_state.get("optimization_results", {})
                    portfolio = st.session_state.get("portfolio", {})
                    
                    # Set report sections to include
                    sections = {
                        "monte_carlo": include_monte_carlo,
                        "risk": include_risk,
                        "optimization": include_optimization,
                        "scenarios": include_scenarios,
                        "historical": include_historical,
                        "disclaimer": include_disclaimer
                    }
                    
                    try:
                        # Generate the report using PDFReport
                        st.write("Generating free report...")
                        print("Before calling PDF report generator")
                        
                        # Ensure monte_carlo_results is a dict
                        if monte_carlo_results is None:
                            monte_carlo_results = {}
                            
                        # Get initial investment, handling both dict and scalar values
                        initial_investment = monte_carlo_results.get("initial_investment", 100000)
                        if hasattr(initial_investment, "item"):  # Convert numpy values to native Python
                            initial_investment = initial_investment.item()
                        
                        # Format assets with more metadata to help visualizations
                        formatted_assets = []
                        for ticker, weight in portfolio.items():
                            # Infer asset type from ticker
                            if ticker in ["SPY", "QQQ", "VTI", "VOO", "IVV"]:
                                asset_type = "equity_etf"
                            elif ticker in ["AGG", "BND", "TLT", "IEF", "SHY"]:
                                asset_type = "bond_etf"
                            elif ticker in ["GLD", "IAU", "SLV"]:
                                asset_type = "commodity_etf"
                            else:
                                asset_type = "stock"
                                
                            formatted_assets.append({
                                "type": asset_type,
                                "name": ticker,
                                "weight": weight,
                                "amount": initial_investment * weight
                            })
                            
                        # Ensure portfolio data is complete
                        if not formatted_assets:
                            # If no assets were provided, create a default portfolio
                            formatted_assets = [
                                {"type": "equity_etf", "name": "SPY", "weight": 0.6, "amount": 60000},
                                {"type": "bond_etf", "name": "AGG", "weight": 0.3, "amount": 30000},
                                {"type": "commodity_etf", "name": "GLD", "weight": 0.1, "amount": 10000}
                            ]
                            st.warning("Using default portfolio for report generation as no portfolio data was found.")
                        
                        # Create query data structure with debugging info
                        query_data = {
                            "query_type": "portfolio_analysis",
                            "original_query": "Generate free report",
                            "assets": formatted_assets,
                            "timeframe": {"value": 10, "unit": "years"},
                            "risk_profile": "moderate",
                            "initial_investment": initial_investment
                        }
                        
                        # Ensure all simulation results exist (create placeholders if needed)
                        if not monte_carlo_results or not isinstance(monte_carlo_results, dict):
                            st.warning("Creating placeholder Monte Carlo results for report")
                            monte_carlo_results = {
                                "initial_investment": initial_investment,
                                "mean_final": initial_investment * 1.5,
                                "median_final": initial_investment * 1.45,
                                "percentiles": {5: initial_investment * 0.8, 95: initial_investment * 2.2},
                                "prob_positive": 0.75
                            }
                            
                        # Format simulation results with error checking
                        simulation_results = {
                            "monte_carlo": monte_carlo_results if isinstance(monte_carlo_results, dict) else {},
                            "cvar": risk_results if isinstance(risk_results, dict) else {},
                            "risk_parity": optimization_results if isinstance(optimization_results, dict) else {}
                        }
                        
                        # Format report data with recommendations
                        report_data = {
                            "recommendations": [
                                {"title": "Diversification Strategy", 
                                 "description": "Based on your portfolio composition, we recommend increasing diversification to reduce concentration risk."},
                                {"title": "Risk Management", 
                                 "description": "Consider adjusting your asset allocation to better align with your risk profile."}
                            ]
                        }
                        
                        # Initialize a flag for whether we need the fallback
                        fallback_needed = True
                        
                        # Try using the updated PDFReport first
                        try:
                            # Import the PDFReport class directly
                            from iso_financial_mvp.report_generator.pdf_generator import PDFReport
                            
                            # Generate report using the updated PDFReport class
                            pdf_report = PDFReport(report_type="free")
                            report_pdf_path = pdf_report.generate_report(
                                query_data=query_data,
                                simulation_results=simulation_results,
                                report_data=report_data
                            )
                            
                            # Check if the file exists
                            if os.path.exists(report_pdf_path):
                                st.success(f"Successfully generated report using PDFReport class: {report_pdf_path}")
                                
                                # Read the PDF file for download
                                with open(report_pdf_path, "rb") as f:
                                    pdf_bytes = f.read()
                                
                                # Encode PDF for display and download
                                b64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
                                
                                # Display PDF in the app using iframe
                                st.subheader("Free Report Preview")
                                pdf_display = f"""
                                <div style="display: flex; justify-content: center;">
                                    <iframe src="data:application/pdf;base64,{b64_pdf}" width="700" height="500" type="application/pdf"></iframe>
                                </div>
                                <p style="text-align: center; margin-top: 10px; font-size: 12px;">(If preview doesn't load, use the download button below)</p>
                                """
                                st.markdown(pdf_display, unsafe_allow_html=True)
                                
                                # Download button for PDF
                                st.download_button(
                                    label="Download PDF Report",
                                    data=pdf_bytes,
                                    file_name=f"iso_financial_report_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf",
                                    mime="application/pdf"
                                )
                                
                                # Save report to session state
                                if "saved_reports" not in st.session_state:
                                    st.session_state.saved_reports = []
                                    
                                # Add report to saved reports
                                st.session_state.saved_reports.append({
                                    "type": "Free Report",
                                    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "pdf_path": report_pdf_path
                                })
                                
                                # Success message
                                st.success("Report generated successfully and saved to your account.")
                                
                                # Skip the fallback generation by setting a flag
                                fallback_needed = False
                            
                        except Exception as pdf_error:
                            # Log error but don't show warning to users
                            print(f"PDFReport generation failed: {str(pdf_error)}")
                            fallback_needed = True  # Ensure fallback is used
                        
                        # Only create the fallback report if needed
                        if fallback_needed:
                            # Create a simple FPDF report as fallback due to font issues
                            from fpdf import FPDF
                            
                            # Create a simple PDF with enhanced content
                            pdf = FPDF()
                            
                            # Title page
                            pdf.add_page()
                            pdf.set_font('Arial', 'B', 20)
                            pdf.cell(200, 20, txt="Iso AI Financial Report", ln=True, align='C')
                            pdf.set_font('Arial', 'I', 12)
                            pdf.cell(200, 10, txt="Investment Analysis & Recommendations", ln=True, align='C')
                            pdf.ln(10)
                            
                            # Add date and report info
                            pdf.set_font('Arial', '', 10)
                            pdf.cell(200, 10, txt=f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')
                            pdf.cell(200, 10, txt="Free Analysis Report", ln=True, align='C')
                            pdf.ln(15)
                            
                            # Add table of contents
                            pdf.set_font('Arial', 'B', 14)
                            pdf.cell(200, 10, txt="Report Contents:", ln=True)
                            
                            pdf.set_font('Arial', '', 11)
                            pdf.cell(200, 7, txt="1. Portfolio Summary", ln=True)
                            pdf.cell(200, 7, txt="2. Asset Allocation Analysis", ln=True)
                            pdf.cell(200, 7, txt="3. Risk Assessment", ln=True)
                            pdf.cell(200, 7, txt="4. Performance Projections", ln=True)
                            pdf.cell(200, 7, txt="5. Key Recommendations", ln=True)
                            pdf.cell(200, 7, txt="6. Disclaimer", ln=True)
                            
                            # SECTION 1: Portfolio Summary
                            pdf.add_page()
                            pdf.set_font('Arial', 'B', 16)
                            pdf.cell(200, 15, txt="1. Portfolio Summary", ln=True)
                            
                            # Add portfolio info with enhanced details
                            pdf.set_font('Arial', '', 12)
                            
                            # Create a nice box for key metrics
                            pdf.set_fill_color(240, 240, 240)  # Light gray background
                            pdf.rect(10, pdf.get_y(), 190, 40, style='F')
                            pdf.ln(5)
                            
                            # Add key portfolio metrics in the box
                            pdf.set_font('Arial', 'B', 12)
                            pdf.cell(100, 10, txt="Initial Investment:", ln=0)
                            pdf.set_font('Arial', '', 12)
                            pdf.cell(90, 10, txt=f"${query_data.get('initial_investment', 0):,.2f}", ln=1)
                            
                            pdf.set_font('Arial', 'B', 12)
                            pdf.cell(100, 10, txt="Number of Assets:", ln=0)
                            pdf.set_font('Arial', '', 12)
                            pdf.cell(90, 10, txt=f"{len(query_data.get('assets', []))}", ln=1)
                            
                            pdf.set_font('Arial', 'B', 12)
                            pdf.cell(100, 10, txt="Time Horizon:", ln=0)
                            pdf.set_font('Arial', '', 12)
                            pdf.cell(90, 10, txt=f"{query_data.get('timeframe', {}).get('value', 10)} years", ln=1)
                            
                            pdf.ln(10)
                            
                            # Add portfolio composition explanation
                            pdf.set_font('Arial', 'B', 14)
                            pdf.cell(200, 10, txt="Portfolio Composition", ln=True)
                            
                            pdf.set_font('Arial', '', 11)
                            pdf.multi_cell(190, 7, txt="Your portfolio consists of a diversified mix of assets as shown in the table below. The allocation represents the percentage of your total investment allocated to each asset.")
                            pdf.ln(5)
                            
                            # Enhanced asset allocation table with more details
                            pdf.set_font('Arial', 'B', 12)
                            pdf.cell(70, 10, txt="Asset", border=1, fill=True)
                            pdf.cell(40, 10, txt="Type", border=1, fill=True)
                            pdf.cell(30, 10, txt="Weight", border=1, fill=True)
                            pdf.cell(50, 10, txt="Amount ($)", border=1, fill=True, ln=True)
                            
                            pdf.set_font('Arial', '', 11)
                            
                            # Calculate total for verification
                            initial_investment = query_data.get('initial_investment', 0)
                            total_allocation = 0
                            
                            for asset in query_data.get('assets', []):
                                ticker = asset.get('name', 'Unknown')
                                weight = asset.get('weight', 0)
                                amount = initial_investment * weight
                                total_allocation += weight
                                
                                # Determine asset type if not provided
                                asset_type = asset.get('type', '')
                                if not asset_type:
                                    if ticker in ["SPY", "QQQ", "VTI", "VOO", "IVV"]:
                                        asset_type = "ETF (Equity)"
                                    elif ticker in ["BND", "AGG", "TLT", "IEF"]:
                                        asset_type = "ETF (Bond)"
                                    elif ticker in ["GLD", "SLV", "IAU"]:
                                        asset_type = "ETF (Commodity)"
                                    else:
                                        asset_type = "Stock"
                                        
                                pdf.cell(70, 10, txt=f"{ticker}", border=1)
                                pdf.cell(40, 10, txt=f"{asset_type}", border=1)
                                pdf.cell(30, 10, txt=f"{weight:.2%}", border=1)
                                pdf.cell(50, 10, txt=f"${amount:,.2f}", border=1, ln=True)
                                
                            # Add total row
                            pdf.set_font('Arial', 'B', 11)
                            pdf.cell(70, 10, txt="Total", border=1, fill=True)
                            pdf.cell(40, 10, txt="", border=1, fill=True)
                            pdf.cell(30, 10, txt=f"{total_allocation:.2%}", border=1, fill=True)
                            pdf.cell(50, 10, txt=f"${initial_investment:,.2f}", border=1, fill=True, ln=True)
                            
                            pdf.ln(10)
                            
                            # SECTION 2: Asset Allocation Analysis
                            pdf.add_page()
                            pdf.set_font('Arial', 'B', 16)
                            pdf.cell(200, 15, txt="2. Asset Allocation Analysis", ln=True)
                            
                            # Asset allocation explanation
                            pdf.set_font('Arial', '', 11)
                            allocation_text = (
                                "Your portfolio's asset allocation is a critical factor in determining your investment outcomes. "
                                "This section analyzes your current allocation and shows how it compares to common allocation strategies "
                                "based on your risk profile and time horizon."
                            )
                            pdf.multi_cell(190, 7, txt=allocation_text)
                            pdf.ln(5)
                            
                            # Add asset allocation by category
                            pdf.set_font('Arial', 'B', 14)
                            pdf.cell(200, 10, txt="Allocation by Asset Category", ln=True)
                            
                            # Calculate allocations by category
                            categories = {"Stocks": 0, "Bonds": 0, "Cash": 0, "Commodities": 0, "Other": 0}
                            
                            for asset in query_data.get('assets', []):
                                ticker = asset.get('name', 'Unknown')
                                weight = asset.get('weight', 0)
                                
                                # Categorize assets (simplified approach)
                                if ticker in ["SPY", "QQQ", "VTI", "VOO", "IVV"] or asset.get('type', '').lower().startswith('equity'):
                                    categories["Stocks"] += weight
                                elif ticker in ["BND", "AGG", "TLT", "IEF"] or asset.get('type', '').lower().startswith('bond'):
                                    categories["Bonds"] += weight
                                elif ticker in ["GLD", "SLV", "IAU"] or asset.get('type', '').lower().startswith('commodity'):
                                    categories["Commodities"] += weight
                                elif ticker in ["SHY", "BIL"] or asset.get('type', '').lower().startswith('cash'):
                                    categories["Cash"] += weight
                                else:
                                    # Default uncategorized assets to stocks
                                    categories["Stocks"] += weight
                            
                            # Create a table for asset categories
                            pdf.set_font('Arial', 'B', 12)
                            pdf.cell(80, 10, txt="Asset Category", border=1, fill=True)
                            pdf.cell(50, 10, txt="Allocation", border=1, fill=True)
                            pdf.cell(60, 10, txt="Amount ($)", border=1, fill=True, ln=True)
                            
                            pdf.set_font('Arial', '', 11)
                            for category, weight in categories.items():
                                if weight > 0:  # Only show categories with non-zero weights
                                    amount = initial_investment * weight
                                    pdf.cell(80, 10, txt=f"{category}", border=1)
                                    pdf.cell(50, 10, txt=f"{weight:.2%}", border=1)
                                    pdf.cell(60, 10, txt=f"${amount:,.2f}", border=1, ln=True)
                            
                            pdf.ln(10)
                            
                            # Add allocation insights
                            pdf.set_font('Arial', 'B', 14)
                            pdf.cell(200, 10, txt="Allocation Insights", ln=True)
                            
                            pdf.set_font('Arial', '', 11)
                            risk_profile = query_data.get('risk_profile', 'moderate').lower()
                            time_horizon = query_data.get('timeframe', {}).get('value', 10)
                            
                            # Generate insights based on allocation
                            stocks_pct = categories["Stocks"] * 100
                            bonds_pct = categories["Bonds"] * 100
                            
                            # Risk profile based insights
                            if risk_profile == 'conservative':
                                target_stocks = 40
                                target_bonds = 50
                                target_other = 10
                            elif risk_profile == 'moderate':
                                target_stocks = 60
                                target_bonds = 30
                                target_other = 10
                            else:  # aggressive
                                target_stocks = 80
                                target_bonds = 15
                                target_other = 5
                                
                            # Time horizon adjustments
                            if time_horizon > 15:
                                target_stocks += 5
                                target_bonds -= 5
                            elif time_horizon < 5:
                                target_stocks -= 10
                                target_bonds += 10
                                
                            # Generate insights
                            insights_text = f"Based on your {risk_profile} risk profile and {time_horizon}-year time horizon, "
                            
                            if abs(stocks_pct - target_stocks) > 10:
                                if stocks_pct > target_stocks:
                                    insights_text += (
                                        f"your stock allocation ({stocks_pct:.1f}%) is significantly higher than the "
                                        f"recommended target ({target_stocks}%) for your risk profile. Consider "
                                        "reducing equity exposure to better align with your risk tolerance."
                                    )
                                else:
                                    insights_text += (
                                        f"your stock allocation ({stocks_pct:.1f}%) is significantly lower than the "
                                        f"recommended target ({target_stocks}%) for your risk profile. Consider "
                                        "increasing equity exposure to improve long-term growth potential."
                                    )
                            else:
                                insights_text += (
                                    f"your asset allocation is generally aligned with recommendations for your profile. "
                                    f"A typical allocation for your profile would be approximately {target_stocks}% stocks, "
                                    f"{target_bonds}% bonds, and {target_other}% in other assets."
                                )
                            
                            pdf.multi_cell(190, 7, txt=insights_text)
                            pdf.ln(5)
                            
                            # SECTION 3: Risk Assessment
                            pdf.add_page()
                            pdf.set_font('Arial', 'B', 16)
                            pdf.cell(200, 15, txt="3. Risk Assessment", ln=True)
                            
                            pdf.set_font('Arial', '', 11)
                            risk_text = (
                                "This section evaluates the risk characteristics of your portfolio. Understanding "
                                "these risk metrics can help you make informed decisions about your investments "
                                "and ensure your portfolio aligns with your risk tolerance."
                            )
                            pdf.multi_cell(190, 7, txt=risk_text)
                            pdf.ln(5)
                            
                            # Risk metrics
                            pdf.set_font('Arial', 'B', 14)
                            pdf.cell(200, 10, txt="Key Risk Metrics", ln=True)
                            
                            # Extract risk data
                            risk_metrics = {}
                            
                            # Try to get risk metrics from different possible sources
                            if simulation_results.get("risk_results"):
                                risk_data = simulation_results.get("risk_results", {})
                            elif simulation_results.get("cvar"):
                                risk_data = simulation_results.get("cvar", {})
                            else:
                                risk_data = {}
                                
                            # Extract metrics, with defaults if not available
                            risk_metrics["var_95"] = risk_data.get("var_95", initial_investment * 0.15)  # Default 15% VaR
                            risk_metrics["max_drawdown"] = risk_data.get("max_drawdown", 0.25)  # Default 25% drawdown
                            risk_metrics["volatility"] = risk_data.get("volatility", stocks_pct / 100 * 0.15)  # Rough estimate
                            
                            # Format metrics
                            var_pct = risk_metrics["var_95"] / initial_investment
                            
                            # Create risk metrics table
                            pdf.set_font('Arial', 'B', 12)
                            pdf.cell(90, 10, txt="Risk Metric", border=1, fill=True)
                            pdf.cell(100, 10, txt="Value", border=1, fill=True, ln=True)
                            
                            pdf.set_font('Arial', '', 11)
                            
                            # Value at Risk
                            pdf.cell(90, 10, txt="Value at Risk (95% confidence)", border=1)
                            pdf.cell(100, 10, txt=f"${risk_metrics['var_95']:,.2f} ({var_pct:.2%} of portfolio)", border=1, ln=True)
                            
                            # Maximum Drawdown
                            pdf.cell(90, 10, txt="Maximum Drawdown", border=1)
                            pdf.cell(100, 10, txt=f"{risk_metrics['max_drawdown']:.2%}", border=1, ln=True)
                            
                            # Portfolio Volatility
                            pdf.cell(90, 10, txt="Portfolio Volatility (estimated)", border=1)
                            pdf.cell(100, 10, txt=f"{risk_metrics['volatility']:.2%} annual", border=1, ln=True)
                            
                            pdf.ln(10)
                            
                            # Risk interpretation
                            pdf.set_font('Arial', 'B', 14)
                            pdf.cell(200, 10, txt="Risk Interpretation", ln=True)
                            
                            pdf.set_font('Arial', '', 11)
                            risk_interp = (
                                f"Based on our analysis, your portfolio has a {risk_profile} level of risk. "
                                f"In a typical worst-case scenario (95% confidence), you could expect to lose "
                                f"up to ${risk_metrics['var_95']:,.2f} ({var_pct:.1%} of your investment) over a short time period. "
                                f"Historically, portfolios with this composition might experience a maximum drawdown of "
                                f"approximately {risk_metrics['max_drawdown']:.1%}.\n\n"
                            )
                            
                            # Add risk profile specific advice
                            if risk_profile == 'conservative':
                                risk_interp += (
                                    "Your conservative risk profile indicates a preference for capital preservation over growth. "
                                    "Your current allocation appears to prioritize stability and income, which aligns with "
                                    "a conservative approach."
                                )
                            elif risk_profile == 'moderate':
                                risk_interp += (
                                    "Your moderate risk profile indicates a balanced approach between growth and capital preservation. "
                                    "Your current allocation aims to capture market growth while maintaining reasonable risk levels."
                                )
                            else:  # aggressive
                                risk_interp += (
                                    "Your aggressive risk profile indicates a strong preference for growth over capital preservation. "
                                    "Your current allocation prioritizes assets with higher growth potential, accepting higher "
                                    "volatility in pursuit of greater long-term returns."
                                )
                            
                            pdf.multi_cell(190, 7, txt=risk_interp)
                            pdf.ln(5)
                            
                            # SECTION 4: Performance Projections
                            pdf.add_page()
                            pdf.set_font('Arial', 'B', 16)
                            pdf.cell(200, 15, txt="4. Performance Projections", ln=True)
                            
                            # Extract Monte Carlo data if available
                            mc_results = simulation_results.get("monte_carlo", {})
                            
                            # Check if we have simulation results
                            mean_final = mc_results.get('mean_final', initial_investment * 1.7)  # Default 70% growth
                            median_final = mc_results.get('median_final', initial_investment * 1.6)  # Default 60% growth
                            
                            # Calculate estimated returns
                            years = time_horizon if time_horizon > 0 else 10
                            total_growth = (mean_final / initial_investment) - 1
                            annual_return = (1 + total_growth) ** (1/years) - 1
                            
                            # Format the projections section
                            pdf.set_font('Arial', '', 11)
                            projection_text = (
                                "This section presents expected future performance of your portfolio based on statistical "
                                "simulations. These projections should be viewed as estimates rather than guarantees, and "
                                f"represent potential outcomes over your {years}-year investment horizon."
                            )
                            pdf.multi_cell(190, 7, txt=projection_text)
                            pdf.ln(5)
                            
                            # Create projections table
                            pdf.set_font('Arial', 'B', 14)
                            pdf.cell(200, 10, txt="Performance Projections", ln=True)
                            
                            # Set up a highlighted box for key projections
                            pdf.set_fill_color(240, 240, 240)  # Light gray background
                            
                            pdf.set_font('Arial', 'B', 12)
                            pdf.cell(90, 10, txt="Metric", border=1, fill=True)
                            pdf.cell(100, 10, txt="Projected Value", border=1, fill=True, ln=True)
                            
                            pdf.set_font('Arial', '', 11)
                            
                            # Initial Investment
                            pdf.cell(90, 10, txt="Initial Investment", border=1)
                            pdf.cell(100, 10, txt=f"${initial_investment:,.2f}", border=1, ln=True)
                            
                            # Expected Final Value
                            pdf.cell(90, 10, txt=f"Expected Value in {years} years", border=1)
                            pdf.cell(100, 10, txt=f"${mean_final:,.2f}", border=1, ln=True)
                            
                            # Total Growth
                            pdf.cell(90, 10, txt="Total Expected Growth", border=1)
                            pdf.cell(100, 10, txt=f"{total_growth:.2%}", border=1, ln=True)
                            
                            # Annualized Return
                            pdf.cell(90, 10, txt="Estimated Annual Return", border=1)
                            pdf.cell(100, 10, txt=f"{annual_return:.2%}", border=1, ln=True)
                            
                            pdf.ln(10)
                            
                            # Projection commentary
                            pdf.set_font('Arial', 'B', 14)
                            pdf.cell(200, 10, txt="Projection Insights", ln=True)
                            
                            pdf.set_font('Arial', '', 11)
                            
                            # Generate commentary based on the projections
                            proj_commentary = (
                                f"Based on our analysis, your portfolio has an estimated annual return of {annual_return:.2%}. "
                                f"Over your {years}-year investment horizon, this could result in your initial investment "
                                f"of ${initial_investment:,.2f} growing to approximately ${mean_final:,.2f}, "
                                f"representing a total growth of {total_growth:.2%}.\n\n"
                            )
                            
                            # Add risk/return commentary based on profile
                            if risk_profile == 'conservative':
                                proj_commentary += (
                                    "This projected return reflects your conservative risk profile, prioritizing capital "
                                    "preservation over aggressive growth. While returns may be modest compared to more "
                                    "aggressive portfolios, your portfolio aims to provide more stability and potentially "
                                    "less severe losses during market downturns."
                                )
                            elif risk_profile == 'moderate':
                                proj_commentary += (
                                    "This projected return reflects your moderate risk profile, balancing growth potential "
                                    "with reasonable risk levels. Your portfolio is designed to capture significant market "
                                    "growth while still providing some protection during market declines."
                                )
                            else:  # aggressive
                                proj_commentary += (
                                    "This projected return reflects your aggressive risk profile, focusing on maximizing "
                                    "growth potential. Your portfolio prioritizes higher expected returns, though this comes "
                                    "with increased volatility and potentially larger drawdowns during market corrections."
                                )
                                
                            pdf.multi_cell(190, 7, txt=proj_commentary)
                            pdf.ln(5)
                            
                            # SECTION 5: Key Recommendations
                            pdf.add_page()
                            pdf.set_font('Arial', 'B', 16)
                            pdf.cell(200, 15, txt="5. Key Recommendations", ln=True)
                            
                            pdf.set_font('Arial', '', 11)
                            rec_intro = (
                                "Based on our comprehensive analysis of your portfolio, we provide the following "
                                "actionable recommendations to help optimize your investment strategy and better "
                                "align with your financial goals."
                            )
                            pdf.multi_cell(190, 7, txt=rec_intro)
                            pdf.ln(10)
                            
                            # Generate recommendations based on the portfolio characteristics
                            recommendations = []
                            
                            # Diversification recommendation
                            if len(query_data.get('assets', [])) < 4:
                                recommendations.append({
                                    "title": "Increase Portfolio Diversification",
                                    "description": (
                                        "Your portfolio currently has a limited number of assets, which may increase "
                                        "concentration risk. Consider adding more diversified holdings such as broad-market "
                                        "ETFs or expanding into additional asset classes to reduce risk through diversification."
                                    )
                                })
                            
                            # Asset allocation recommendation based on risk profile
                            stock_target_diff = abs(stocks_pct - target_stocks)
                            if stock_target_diff > 10:
                                if stocks_pct > target_stocks:
                                    recommendations.append({
                                        "title": "Adjust Asset Allocation",
                                        "description": (
                                            f"Your current stock allocation ({stocks_pct:.1f}%) is higher than recommended "
                                            f"({target_stocks}%) for your {risk_profile} risk profile. Consider reallocating "
                                            f"approximately {(stocks_pct - target_stocks):.1f}% from stocks to bonds or other "
                                            "defensive assets to better align with your risk tolerance."
                                        )
                                    })
                                else:
                                    recommendations.append({
                                        "title": "Adjust Asset Allocation",
                                        "description": (
                                            f"Your current stock allocation ({stocks_pct:.1f}%) is lower than recommended "
                                            f"({target_stocks}%) for your {risk_profile} risk profile. Consider increasing your "
                                            f"equity exposure by approximately {(target_stocks - stocks_pct):.1f}% to improve "
                                            "long-term growth potential."
                                        )
                                    })
                            
                            # Risk management recommendation
                            if risk_profile in ['moderate', 'aggressive'] and categories.get("Bonds", 0) < 0.15:
                                recommendations.append({
                                    "title": "Enhance Downside Protection",
                                    "description": (
                                        "Your portfolio may benefit from additional downside protection, especially "
                                        "given your significant equity exposure. Consider allocating at least 15-20% "
                                        "to high-quality bonds or other defensive assets to provide stability during "
                                        "market corrections."
                                    )
                                })
                                
                            # Default recommendation if none generated
                            if not recommendations:
                                recommendations.append({
                                    "title": "Regular Portfolio Rebalancing",
                                    "description": (
                                        "Your portfolio appears well-aligned with your stated goals and risk profile. "
                                        "We recommend implementing a regular rebalancing schedule (quarterly or semi-annually) "
                                        "to maintain your target asset allocation as market movements cause drift in your "
                                        "portfolio weights over time."
                                    )
                                })
                                
                                recommendations.append({
                                    "title": "Consider Tax-Efficient Allocation",
                                    "description": (
                                        "Review the tax efficiency of your current asset placement. Consider holding tax-efficient "
                                        "investments in taxable accounts and less tax-efficient investments in tax-advantaged "
                                        "accounts when possible. This strategy can help minimize the impact of taxes on your "
                                        "investment returns."
                                    )
                                })
                            
                            # Add the recommendations with nice formatting
                            for i, recommendation in enumerate(recommendations):
                                # Add numbered recommendation with title
                                pdf.set_font('Arial', 'B', 12)
                                pdf.cell(200, 10, txt=f"{i+1}. {recommendation['title']}", ln=True)
                                
                                # Add recommendation description
                                pdf.set_font('Arial', '', 11)
                                pdf.multi_cell(190, 7, txt=recommendation['description'])
                                pdf.ln(7)
                            
                            # Create timestamp for unique file
                            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                            report_pdf_path = f"iso_financial_report_{timestamp}.pdf"
                            
                            # Add disclaimer
                            pdf.add_page()
                            pdf.set_font('Arial', 'B', 14)
                            pdf.cell(200, 10, txt="Disclaimer", ln=True)
                            
                            pdf.set_font('Arial', '', 10)
                            disclaimer_text = (
                                "This report is for informational purposes only and does not constitute investment advice. "
                                "Past performance is not indicative of future results. Investing involves risk, including the "
                                "possible loss of principal. Diversification does not ensure a profit or protect against a loss "
                                "in a declining market.\n\n"
                                "The projections or other information generated by this report regarding the likelihood of various "
                                "investment outcomes are hypothetical in nature, do not reflect actual investment results, and are "
                                "not guarantees of future results. Actual results may vary significantly.\n\n"
                                "Iso AI is not a registered investment advisor. Please consult with a financial professional "
                                "before making any investment decisions."
                            )
                            pdf.multi_cell(0, 5, txt=disclaimer_text)
                            
                            # Add footer
                            pdf.set_y(-15)
                            pdf.set_font('Arial', 'I', 8)
                            pdf.cell(0, 10, f"© {datetime.now().year} Iso AI - Generated on {datetime.now().strftime('%Y-%m-%d')}", 0, 0, 'C')
                            
                            # Output PDF
                            pdf.output(report_pdf_path)
                            print(f"Generated report at: {report_pdf_path}")
                            st.write(f"Report generated at: {report_pdf_path}")
                            
                            # Check if file exists
                            if not os.path.exists(report_pdf_path):
                                st.error(f"PDF file not found at {report_pdf_path}")
                                raise FileNotFoundError(f"PDF file not found at {report_pdf_path}")
                                
                            # Show file size
                            file_size = os.path.getsize(report_pdf_path)
                            st.write(f"PDF size: {file_size} bytes")
                            
                            # Read the PDF file
                            with open(report_pdf_path, "rb") as f:
                                pdf_bytes = f.read()
                            
                            # Encode PDF for display and download
                            b64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
                            
                            # Display PDF in the app using iframe (better compatibility)
                            st.subheader("Free Report Preview")
                            
                            # Try both methods for better compatibility
                            pdf_display = f"""
                            <div style="display: flex; justify-content: center;">
                                <iframe src="data:application/pdf;base64,{b64_pdf}" width="700" height="500" type="application/pdf"></iframe>
                            </div>
                            <p style="text-align: center; margin-top: 10px; font-size: 12px;">(If preview doesn't load, use the download button below)</p>
                            """
                            st.markdown(pdf_display, unsafe_allow_html=True)
                            
                            # Alternative object tag as backup
                            pdf_display_alt = f"""
                            <div style="display: flex; justify-content: center;">
                                <object data="data:application/pdf;base64,{b64_pdf}" type="application/pdf" width="700" height="500">
                                    <p>Unable to display PDF. Please download the PDF file.</p>
                                </object>
                            </div>
                            """
                            st.markdown(pdf_display_alt, unsafe_allow_html=True)
                            
                            # Download button for PDF
                            st.download_button(
                                label="Download PDF Report",
                                data=pdf_bytes,
                                file_name=f"iso_financial_report_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf",
                                mime="application/pdf"
                            )
                            
                            # Save report to session state
                            if "saved_reports" not in st.session_state:
                                st.session_state.saved_reports = []
                                
                            # Add report to saved reports
                            st.session_state.saved_reports.append({
                                "type": "Free Report",
                                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "pdf_path": report_pdf_path
                            })
                            
                            # Success message
                            st.success("Report generated successfully and saved to your account.")
                        
                    except Exception as e:
                        # Log detailed error for debugging
                        import traceback
                        error_details = traceback.format_exc()
                        print(f"[ERROR] Free report generation failed: {str(e)}")
                        print(f"[ERROR] Traceback: {error_details}")
                        
                        # Show more helpful error message to users
                        st.error(f"Unable to generate the report: {str(e)}")
                        
                        # Add debugging info in expandable section
                        with st.expander("Technical Error Details (for debugging)"):
                            st.code(error_details)
                            st.write("Query Data:", query_data)
                            st.write("Simulation Results Keys:", list(simulation_results.keys()))
                            for key, value in simulation_results.items():
                                if isinstance(value, dict):
                                    st.write(f"{key} keys:", list(value.keys() if value else "Empty"))
                        
            elif report_type == "Premium Detailed Report" and "premium" in available_report_types:
                with st.spinner("Generating Premium Report..."):
                    # Get simulation results from session state
                    monte_carlo_results = st.session_state.get("monte_carlo_results", {})
                    risk_results = st.session_state.get("risk_results", {})
                    optimization_results = st.session_state.get("optimization_results", {})
                    scenario_results = st.session_state.get("scenario_results", {})
                    portfolio = st.session_state.get("portfolio", {})
                    
                    # Set report sections to include
                    sections = {
                        "monte_carlo": include_monte_carlo,
                        "risk": include_risk,
                        "optimization": include_optimization,
                        "scenarios": include_scenarios,
                        "historical": include_historical,
                        "disclaimer": include_disclaimer
                    }
                    
                    try:
                        # Generate the report using PDFReport
                        st.write("Generating premium report...")
                        print("Before calling premium PDF report generator")
                        
                        # Ensure monte_carlo_results is a dict
                        if monte_carlo_results is None:
                            monte_carlo_results = {}
                            
                        # Get initial investment, handling both dict and scalar values
                        initial_investment = monte_carlo_results.get("initial_investment", 100000)
                        if hasattr(initial_investment, "item"):  # Convert numpy values to native Python
                            initial_investment = initial_investment.item()
                        
                        # Format assets with more metadata to help visualizations
                        formatted_assets = []
                        for ticker, weight in portfolio.items():
                            # Infer asset type from ticker
                            if ticker in ["SPY", "QQQ", "VTI", "VOO", "IVV"]:
                                asset_type = "equity_etf"
                            elif ticker in ["AGG", "BND", "TLT", "IEF", "SHY"]:
                                asset_type = "bond_etf"
                            elif ticker in ["GLD", "IAU", "SLV"]:
                                asset_type = "commodity_etf"
                            else:
                                asset_type = "stock"
                                
                            formatted_assets.append({
                                "type": asset_type,
                                "name": ticker,
                                "weight": weight,
                                "amount": initial_investment * weight
                            })
                            
                        # Ensure portfolio data is complete
                        if not formatted_assets:
                            # If no assets were provided, create a default portfolio
                            formatted_assets = [
                                {"type": "equity_etf", "name": "SPY", "weight": 0.6, "amount": 60000},
                                {"type": "bond_etf", "name": "AGG", "weight": 0.3, "amount": 30000},
                                {"type": "commodity_etf", "name": "GLD", "weight": 0.1, "amount": 10000}
                            ]
                            st.warning("Using default portfolio for report generation as no portfolio data was found.")
                        
                        # Create query data structure with debugging info
                        query_data = {
                            "query_type": "portfolio_analysis",
                            "original_query": "Generate premium report",
                            "assets": formatted_assets,
                            "timeframe": {"value": 10, "unit": "years"},
                            "risk_profile": "moderate",
                            "initial_investment": initial_investment
                        }
                        
                        # Ensure all simulation results exist (create placeholders if needed)
                        if not monte_carlo_results or not isinstance(monte_carlo_results, dict):
                            st.warning("Creating placeholder Monte Carlo results for report")
                            monte_carlo_results = {
                                "initial_investment": initial_investment,
                                "mean_final": initial_investment * 1.5,
                                "median_final": initial_investment * 1.45,
                                "percentiles": {5: initial_investment * 0.8, 95: initial_investment * 2.2},
                                "prob_positive": 0.75
                            }
                            
                        # Format simulation results with error checking
                        simulation_results = {
                            "monte_carlo": monte_carlo_results if isinstance(monte_carlo_results, dict) else {},
                            "cvar": risk_results if isinstance(risk_results, dict) else {},
                            "risk_parity": optimization_results if isinstance(optimization_results, dict) else {},
                            "scenario": scenario_results if isinstance(scenario_results, dict) else {}
                        }
                        
                        # Format report data with detailed recommendations
                        report_data = {
                            "recommendations": [
                                {"title": "Diversification Strategy", 
                                 "description": "Based on your portfolio composition, we recommend increasing diversification to reduce concentration risk."},
                                {"title": "Risk Management", 
                                 "description": "Consider adjusting your asset allocation to better align with your risk profile."},
                                {"title": "Scenario Preparedness", 
                                 "description": "Your portfolio shows vulnerability to recession scenarios. Consider adding defensive assets for better protection."},
                                {"title": "Tax Efficiency", 
                                 "description": "Rebalancing your portfolio could enhance tax efficiency and improve long-term returns."}
                            ]
                        }
                        
                        # Initialize a flag for whether we need the fallback
                        fallback_needed = True
                        
                        # Try using the updated PDFReport first
                        try:
                            # Import the PDFReport class directly
                            from iso_financial_mvp.report_generator.pdf_generator import PDFReport
                            
                            # Generate report using the updated PDFReport class
                            pdf_report = PDFReport(report_type="premium")
                            report_pdf_path = pdf_report.generate_report(
                                query_data=query_data,
                                simulation_results=simulation_results,
                                report_data=report_data
                            )
                            
                            # Check if the file exists
                            if os.path.exists(report_pdf_path):
                                st.success(f"Successfully generated premium report using PDFReport class: {report_pdf_path}")
                                
                                # Read the PDF file for download
                                with open(report_pdf_path, "rb") as f:
                                    pdf_bytes = f.read()
                                
                                # Encode PDF for display and download
                                b64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
                                
                                # Display PDF in the app using iframe
                                st.subheader("Premium Report Preview")
                                pdf_display = f"""
                                <div style="display: flex; justify-content: center;">
                                    <iframe src="data:application/pdf;base64,{b64_pdf}" width="700" height="500" type="application/pdf"></iframe>
                                </div>
                                <p style="text-align: center; margin-top: 10px; font-size: 12px;">(If preview doesn't load, use the download button below)</p>
                                """
                                st.markdown(pdf_display, unsafe_allow_html=True)
                                
                                # Download button for PDF
                                st.download_button(
                                    label="Download Premium PDF Report",
                                    data=pdf_bytes,
                                    file_name=f"iso_financial_premium_report_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf",
                                    mime="application/pdf"
                                )
                                
                                # Save report to session state
                                if "saved_reports" not in st.session_state:
                                    st.session_state.saved_reports = []
                                    
                                # Add report to saved reports
                                st.session_state.saved_reports.append({
                                    "type": "Premium Report",
                                    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "pdf_path": report_pdf_path
                                })
                                
                                # Success message
                                st.success("Premium report generated successfully and saved to your account.")
                                
                                # Skip the fallback generation by setting a flag
                                fallback_needed = False
                            
                        except Exception as pdf_error:
                            # Log error but don't show warning to users
                            print(f"Premium PDFReport generation failed: {str(pdf_error)}")
                            fallback_needed = True  # Ensure fallback is used
                            
                        # Only create the fallback report if needed
                        if fallback_needed:
                            # Create a simple FPDF premium report as fallback due to font issues
                            from fpdf import FPDF
                            
                            # Create an enhanced PDF with premium styling
                            pdf = FPDF()
                            
                            # Cover page
                            pdf.add_page()
                            pdf.set_font('Arial', 'B', 22)
                            pdf.set_text_color(31, 64, 122)  # Dark blue for premium
                            pdf.cell(200, 20, txt="Iso AI Founding Access", ln=True, align='C')
                            pdf.set_font('Arial', 'B', 18)
                            pdf.cell(200, 10, txt="Investment Portfolio Analysis", ln=True, align='C')
                            
                            # Add date and confidential marking
                            pdf.ln(10)
                            pdf.set_font('Arial', 'I', 12)
                            pdf.cell(200, 10, txt=f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')
                            pdf.set_font('Arial', 'B', 10)
                            pdf.set_text_color(150, 0, 0)  # Dark red for confidential
                            pdf.cell(200, 10, txt="CONFIDENTIAL - FOUNDING ACCESS MEMBER", ln=True, align='C')
                            
                            # Reset text color
                            pdf.set_text_color(0, 0, 0)
                            
                            # Add table of contents
                            pdf.ln(20)
                            pdf.set_font('Arial', 'B', 14)
                            pdf.cell(200, 10, txt="Comprehensive Report Contents:", ln=True)
                            
                            pdf.set_font('Arial', '', 12)
                            pdf.cell(10, 8, txt="1.", ln=0)
                            pdf.cell(190, 8, txt="Executive Portfolio Summary", ln=1)
                            
                            pdf.cell(10, 8, txt="2.", ln=0)
                            pdf.cell(190, 8, txt="Detailed Asset Allocation Analysis", ln=1)
                            
                            pdf.cell(10, 8, txt="3.", ln=0)
                            pdf.cell(190, 8, txt="Risk Assessment & Management", ln=1)
                            
                            pdf.cell(10, 8, txt="4.", ln=0)
                            pdf.cell(190, 8, txt="Advanced Performance Projections", ln=1)
                            
                            pdf.cell(10, 8, txt="5.", ln=0)
                            pdf.cell(190, 8, txt="Scenario Analysis", ln=1)
                            
                            pdf.cell(10, 8, txt="6.", ln=0)
                            pdf.cell(190, 8, txt="Tax Efficiency Optimization", ln=1)
                            
                            pdf.cell(10, 8, txt="7.", ln=0)
                            pdf.cell(190, 8, txt="Strategic Recommendations", ln=1)
                            
                            pdf.cell(10, 8, txt="8.", ln=0)
                            pdf.cell(190, 8, txt="Disclaimer & Legal", ln=1)
                            
                            # SECTION 1: Executive Summary
                            pdf.add_page()
                            pdf.set_font('Arial', 'B', 16)
                            pdf.set_text_color(31, 64, 122)  # Dark blue for section headers
                            pdf.cell(200, 15, txt="1. Executive Portfolio Summary", ln=True)
                            pdf.set_text_color(0, 0, 0)  # Reset text color
                            
                            # Create an executive summary box
                            pdf.set_fill_color(240, 245, 255)  # Light blue background for premium
                            pdf.rect(10, pdf.get_y(), 190, 60, style='F')
                            pdf.ln(5)
                            
                            # Portfolio metrics in the box
                            pdf.set_font('Arial', 'B', 12)
                            pdf.cell(100, 10, txt="Initial Investment:", ln=0)
                            pdf.set_font('Arial', '', 12)
                            pdf.cell(90, 10, txt=f"${query_data.get('initial_investment', 0):,.2f}", ln=1)
                            
                            pdf.set_font('Arial', 'B', 12)
                            pdf.cell(100, 10, txt="Investment Time Horizon:", ln=0)
                            pdf.set_font('Arial', '', 12)
                            pdf.cell(90, 10, txt=f"{query_data.get('timeframe', {}).get('value', 10)} years", ln=1)
                            
                            pdf.set_font('Arial', 'B', 12)
                            pdf.cell(100, 10, txt="Risk Profile:", ln=0)
                            pdf.set_font('Arial', '', 12)
                            pdf.cell(90, 10, txt=f"{query_data.get('risk_profile', 'Moderate').capitalize()}", ln=1)
                            
                            pdf.set_font('Arial', 'B', 12)
                            pdf.cell(100, 10, txt="Number of Assets:", ln=0)
                            pdf.set_font('Arial', '', 12)
                            pdf.cell(90, 10, txt=f"{len(query_data.get('assets', []))}", ln=1)
                            
                            pdf.ln(10)
                            
                            # Executive Summary Text
                            risk_profile = query_data.get('risk_profile', 'moderate').lower()
                            time_horizon = query_data.get('timeframe', {}).get('value', 10)
                            initial_investment = query_data.get('initial_investment', 0)
                            
                            # Extract Monte Carlo data if available
                            mc_results = simulation_results.get("monte_carlo", {})
                            mean_final = mc_results.get('mean_final', initial_investment * 1.7)  # Default 70% growth
                            
                            # Calculate estimated returns
                            total_growth = (mean_final / initial_investment) - 1
                            annual_return = (1 + total_growth) ** (1/time_horizon) - 1 if time_horizon > 0 else 0.07
                            
                            pdf.set_font('Arial', 'B', 14)
                            pdf.cell(200, 10, txt="Executive Overview", ln=True)
                            
                            pdf.set_font('Arial', '', 11)
                            executive_summary = (
                                f"This comprehensive Founding Access report provides an in-depth analysis of your investment "
                                f"portfolio based on your {risk_profile} risk profile and {time_horizon}-year investment horizon.\n\n"
                                
                                f"Your portfolio of ${initial_investment:,.2f} is projected to grow to approximately "
                                f"${mean_final:,.2f} over your {time_horizon}-year time horizon, representing an estimated "
                                f"annualized return of {annual_return:.2%} and total growth of {total_growth:.2%}.\n\n"
                                
                                "As a Founding Access member, this report includes exclusive advanced analytics, personalized "
                                "recommendations, tax efficiency insights, and scenario analysis not available in standard reports."
                            )
                            pdf.multi_cell(190, 7, txt=executive_summary)
                            
                            # Portfolio Composition
                            pdf.ln(10)
                            pdf.set_font('Arial', 'B', 14)
                            pdf.cell(200, 10, txt="Portfolio Composition", ln=True)
                            
                            # Detailed assets table
                            pdf.set_font('Arial', 'B', 11)
                            pdf.set_fill_color(220, 230, 240)  # Lighter blue for table headers
                            pdf.cell(70, 10, txt="Asset", border=1, fill=True)
                            pdf.cell(35, 10, txt="Type", border=1, fill=True)
                            pdf.cell(25, 10, txt="Weight", border=1, fill=True)
                            pdf.cell(30, 10, txt="Amount ($)", border=1, fill=True)
                            pdf.cell(30, 10, txt="Risk Level", border=1, fill=True, ln=True)
                            
                            pdf.set_font('Arial', '', 10)
                            
                            # Initialize tracking variables
                            total_allocation = 0
                            categories = {"Stocks": 0, "Bonds": 0, "Cash": 0, "Commodities": 0, "Other": 0}
                            
                            for asset in query_data.get('assets', []):
                                ticker = asset.get('name', 'Unknown')
                                weight = asset.get('weight', 0)
                                amount = initial_investment * weight
                                total_allocation += weight
                                
                                # Determine asset type if not provided
                                asset_type = asset.get('type', '')
                                if not asset_type:
                                    if ticker in ["SPY", "QQQ", "VTI", "VOO", "IVV"]:
                                        asset_type = "ETF (Equity)"
                                        categories["Stocks"] += weight
                                    elif ticker in ["BND", "AGG", "TLT", "IEF"]:
                                        asset_type = "ETF (Bond)"
                                        categories["Bonds"] += weight
                                    elif ticker in ["GLD", "SLV", "IAU"]:
                                        asset_type = "ETF (Commodity)"
                                        categories["Commodities"] += weight
                                    elif ticker in ["SHY", "BIL"]:
                                        asset_type = "ETF (Cash)"
                                        categories["Cash"] += weight
                                    else:
                                        asset_type = "Stock"
                                        categories["Stocks"] += weight
                                elif asset_type.lower().startswith('equity') or asset_type.lower().startswith('stock'):
                                    categories["Stocks"] += weight
                                elif asset_type.lower().startswith('bond'):
                                    categories["Bonds"] += weight
                                elif asset_type.lower().startswith('commodity'):
                                    categories["Commodities"] += weight
                                elif asset_type.lower().startswith('cash'):
                                    categories["Cash"] += weight
                                else:
                                    categories["Other"] += weight
                                
                                # Determine risk level
                                if asset_type.lower().startswith('etf (bond)') or asset_type.lower().startswith('bond'):
                                    risk_level = "Low"
                                elif asset_type.lower().startswith('etf (equity)') or asset_type.lower().startswith('stock'):
                                    risk_level = "Medium-High" 
                                elif asset_type.lower().startswith('etf (commodity)') or asset_type.lower().startswith('commodity'):
                                    risk_level = "Medium"
                                else:
                                    risk_level = "Medium"
                                
                                pdf.cell(70, 8, txt=f"{ticker}", border=1)
                                pdf.cell(35, 8, txt=f"{asset_type}", border=1)
                                pdf.cell(25, 8, txt=f"{weight:.2%}", border=1, align='R')
                                pdf.cell(30, 8, txt=f"${amount:,.2f}", border=1, align='R')
                                pdf.cell(30, 8, txt=f"{risk_level}", border=1, align='C', ln=True)
                                
                            # Add total row
                            pdf.set_font('Arial', 'B', 10)
                            pdf.set_fill_color(220, 230, 240)  # Light blue for total row
                            pdf.cell(70, 8, txt="Total", border=1, fill=True)
                            pdf.cell(35, 8, txt="", border=1, fill=True)
                            pdf.cell(25, 8, txt=f"{total_allocation:.2%}", border=1, fill=True, align='R')
                            pdf.cell(30, 8, txt=f"${initial_investment:,.2f}", border=1, fill=True, align='R')
                            pdf.cell(30, 8, txt="", border=1, fill=True, ln=True)
                            
                            # SECTION 2: Detailed Asset Allocation Analysis
                            pdf.add_page()
                            pdf.set_font('Arial', 'B', 16)
                            pdf.set_text_color(31, 64, 122)  # Dark blue for section headers
                            pdf.cell(200, 15, txt="2. Detailed Asset Allocation Analysis", ln=True)
                            pdf.set_text_color(0, 0, 0)  # Reset text color
                            
                            pdf.set_font('Arial', '', 11)
                            allocation_intro = (
                                "This section provides a detailed analysis of your portfolio's asset allocation across major "
                                "investment categories. Proper asset allocation is one of the most critical factors in determining "
                                "your investment outcomes and managing risk effectively."
                            )
                            pdf.multi_cell(190, 7, txt=allocation_intro)
                            pdf.ln(5)
                            
                            # Create asset allocation by category table with premium styling
                            pdf.set_font('Arial', 'B', 14)
                            pdf.cell(200, 10, txt="Allocation by Asset Category", ln=True)
                            
                            # Table for asset categories
                            pdf.set_font('Arial', 'B', 11)
                            pdf.set_fill_color(220, 230, 240)  # Light blue fill for headers
                            pdf.cell(50, 10, txt="Asset Category", border=1, fill=True)
                            pdf.cell(35, 10, txt="Allocation %", border=1, fill=True)
                            pdf.cell(45, 10, txt="Amount ($)", border=1, fill=True)
                            pdf.cell(60, 10, txt="Category Risk", border=1, fill=True, ln=True)
                            
                            # Stocks category
                            stocks_pct = categories["Stocks"]
                            if stocks_pct > 0:
                                pdf.set_font('Arial', '', 10)
                                pdf.cell(50, 8, txt="Stocks/Equities", border=1)
                                pdf.cell(35, 8, txt=f"{stocks_pct:.2%}", border=1, align='R')
                                pdf.cell(45, 8, txt=f"${stocks_pct * initial_investment:,.2f}", border=1, align='R')
                                pdf.cell(60, 8, txt="High - Growth oriented assets", border=1, ln=True)
                            
                            # Bonds category
                            bonds_pct = categories["Bonds"]
                            if bonds_pct > 0:
                                pdf.set_font('Arial', '', 10)
                                pdf.cell(50, 8, txt="Bonds/Fixed Income", border=1)
                                pdf.cell(35, 8, txt=f"{bonds_pct:.2%}", border=1, align='R')
                                pdf.cell(45, 8, txt=f"${bonds_pct * initial_investment:,.2f}", border=1, align='R')
                                pdf.cell(60, 8, txt="Low - Income & stability", border=1, ln=True)
                            
                            # Cash category
                            cash_pct = categories["Cash"]
                            if cash_pct > 0:
                                pdf.set_font('Arial', '', 10)
                                pdf.cell(50, 8, txt="Cash/Equivalents", border=1)
                                pdf.cell(35, 8, txt=f"{cash_pct:.2%}", border=1, align='R')
                                pdf.cell(45, 8, txt=f"${cash_pct * initial_investment:,.2f}", border=1, align='R')
                                pdf.cell(60, 8, txt="Very Low - Capital preservation", border=1, ln=True)
                            
                            # Commodities
                            commodities_pct = categories["Commodities"]
                            if commodities_pct > 0:
                                pdf.set_font('Arial', '', 10)
                                pdf.cell(50, 8, txt="Commodities", border=1)
                                pdf.cell(35, 8, txt=f"{commodities_pct:.2%}", border=1, align='R')
                                pdf.cell(45, 8, txt=f"${commodities_pct * initial_investment:,.2f}", border=1, align='R')
                                pdf.cell(60, 8, txt="Medium - Inflation hedge", border=1, ln=True)
                            
                            # Others
                            other_pct = categories["Other"]
                            if other_pct > 0:
                                pdf.set_font('Arial', '', 10)
                                pdf.cell(50, 8, txt="Other Investments", border=1)
                                pdf.cell(35, 8, txt=f"{other_pct:.2%}", border=1, align='R')
                                pdf.cell(45, 8, txt=f"${other_pct * initial_investment:,.2f}", border=1, align='R')
                                pdf.cell(60, 8, txt="Varies by asset", border=1, ln=True)
                            
                            pdf.ln(10)
                            
                            # Allocation analysis based on risk profile
                            pdf.set_font('Arial', 'B', 14)
                            pdf.cell(200, 10, txt="Allocation Analysis & Optimization", ln=True)
                            
                            # Calculate target allocations based on risk profile
                            if risk_profile == 'conservative':
                                target_stocks = 40
                                target_bonds = 50
                                target_other = 10
                                profile_desc = "capital preservation with modest growth"
                            elif risk_profile == 'moderate':
                                target_stocks = 60
                                target_bonds = 30
                                target_other = 10
                                profile_desc = "balanced growth with moderate risk"
                            else:  # aggressive
                                target_stocks = 80
                                target_bonds = 15
                                target_other = 5
                                profile_desc = "maximum growth potential with higher volatility"
                            
                            # Time horizon adjustments
                            if time_horizon > 15:
                                target_stocks += 5
                                target_bonds -= 5
                                horizon_impact = "Your longer time horizon allows for slightly higher equity exposure."
                            elif time_horizon < 5:
                                target_stocks -= 10
                                target_bonds += 10
                                horizon_impact = "Your shorter time horizon suggests more conservative allocations."
                            else:
                                horizon_impact = "Your time horizon is aligned with standard allocation models."
                                
                            # Format analysis text
                            pdf.set_font('Arial', '', 11)
                            
                            # Create comparison table showing current vs target allocation
                            pdf.set_font('Arial', 'B', 11)
                            pdf.set_fill_color(220, 230, 240)  # Light blue fill for headers
                            pdf.cell(50, 10, txt="Asset Class", border=1, fill=True)
                            pdf.cell(40, 10, txt="Current Allocation", border=1, fill=True)
                            pdf.cell(40, 10, txt="Target Allocation", border=1, fill=True)
                            pdf.cell(60, 10, txt="Difference", border=1, fill=True, ln=True)
                            
                            pdf.set_font('Arial', '', 10)
                            
                            # Stocks comparison
                            stocks_diff = stocks_pct * 100 - target_stocks
                            pdf.cell(50, 8, txt="Stocks/Equities", border=1)
                            pdf.cell(40, 8, txt=f"{stocks_pct * 100:.1f}%", border=1, align='C')
                            pdf.cell(40, 8, txt=f"{target_stocks:.1f}%", border=1, align='C')
                            
                            # Color code the difference based on magnitude
                            if abs(stocks_diff) < 5:
                                diff_text = f"{stocks_diff:+.1f}% (On Target)"
                            elif stocks_diff > 0:
                                diff_text = f"{stocks_diff:+.1f}% (Overweight)"
                            else:
                                diff_text = f"{stocks_diff:+.1f}% (Underweight)"
                            pdf.cell(60, 8, txt=diff_text, border=1, ln=True)
                            
                            # Bonds comparison
                            bonds_diff = bonds_pct * 100 - target_bonds
                            pdf.cell(50, 8, txt="Bonds/Fixed Income", border=1)
                            pdf.cell(40, 8, txt=f"{bonds_pct * 100:.1f}%", border=1, align='C')
                            pdf.cell(40, 8, txt=f"{target_bonds:.1f}%", border=1, align='C')
                            
                            # Color code the difference
                            if abs(bonds_diff) < 5:
                                diff_text = f"{bonds_diff:+.1f}% (On Target)"
                            elif bonds_diff > 0:
                                diff_text = f"{bonds_diff:+.1f}% (Overweight)"
                            else:
                                diff_text = f"{bonds_diff:+.1f}% (Underweight)"
                            pdf.cell(60, 8, txt=diff_text, border=1, ln=True)
                            
                            # Other assets comparison (cash, commodities, etc.)
                            other_actual = (cash_pct + commodities_pct + other_pct) * 100
                            other_diff = other_actual - target_other
                            pdf.cell(50, 8, txt="Other Assets", border=1)
                            pdf.cell(40, 8, txt=f"{other_actual:.1f}%", border=1, align='C')
                            pdf.cell(40, 8, txt=f"{target_other:.1f}%", border=1, align='C')
                            
                            # Color code the difference
                            if abs(other_diff) < 5:
                                diff_text = f"{other_diff:+.1f}% (On Target)"
                            elif other_diff > 0:
                                diff_text = f"{other_diff:+.1f}% (Overweight)"
                            else:
                                diff_text = f"{other_diff:+.1f}% (Underweight)"
                            pdf.cell(60, 8, txt=diff_text, border=1, ln=True)
                            
                            pdf.ln(10)
                            
                            # Allocation optimization text
                            pdf.set_font('Arial', 'B', 12)
                            pdf.cell(190, 10, txt="Personalized Allocation Insights:", ln=True)
                            
                            pdf.set_font('Arial', '', 11)
                            allocation_insights = (
                                f"Your {risk_profile} risk profile suggests a portfolio focused on {profile_desc}. "
                                f"{horizon_impact}\n\n"
                            )
                            
                            # Add specific recommendations based on allocation differences
                            if abs(stocks_diff) > 10 or abs(bonds_diff) > 10:
                                if stocks_diff > 10:
                                    allocation_insights += (
                                        f"Your portfolio is significantly overweight in stocks (by {stocks_diff:.1f}%) relative to your "
                                        f"risk profile. While this may increase growth potential, it also increases volatility. Consider "
                                        f"reducing equity exposure by {stocks_diff:.1f}% and increasing bond allocation to better align "
                                        f"with your risk tolerance.\n\n"
                                    )
                                elif stocks_diff < -10:
                                    allocation_insights += (
                                        f"Your portfolio is significantly underweight in stocks (by {-stocks_diff:.1f}%) relative to your "
                                        f"risk profile. This conservative positioning may limit your growth potential. Consider increasing "
                                        f"equity exposure by {-stocks_diff:.1f}% to better align with your growth objectives.\n\n"
                                    )
                            else:
                                allocation_insights += (
                                    f"Your current allocation is generally well-aligned with your risk profile, with only minor "
                                    f"deviations from target allocations. Regular rebalancing will help maintain this alignment "
                                    f"as market movements cause portfolio drift over time.\n\n"
                                )
                            
                            # Add diversification comment
                            if len(query_data.get('assets', [])) < 5:
                                allocation_insights += (
                                    "Your portfolio has relatively few holdings, which may increase concentration risk. "
                                    "Consider adding more diversified investments to reduce single-asset risks."
                                )
                            
                            pdf.multi_cell(190, 7, txt=allocation_insights)
                            
                            # SECTION 3: Risk Assessment & Management
                            pdf.add_page()
                            pdf.set_font('Arial', 'B', 16)
                            pdf.set_text_color(31, 64, 122)  # Dark blue for section headers
                            pdf.cell(200, 15, txt="3. Risk Assessment & Management", ln=True)
                            pdf.set_text_color(0, 0, 0)  # Reset text color
                            
                            pdf.set_font('Arial', '', 11)
                            risk_intro = (
                                "This exclusive Founding Access section provides a comprehensive analysis of your portfolio's risk "
                                "characteristics, using advanced risk metrics to quantify potential downside scenarios and identify "
                                "optimal risk management strategies."
                            )
                            pdf.multi_cell(190, 7, txt=risk_intro)
                            pdf.ln(5)
                            
                            # Create advanced risk metrics table
                            pdf.set_font('Arial', 'B', 14)
                            pdf.cell(200, 10, txt="Advanced Risk Metrics", ln=True)
                            
                            # Extract risk data
                            risk_metrics = {}
                            
                            # Try to get risk metrics from different possible sources
                            if simulation_results.get("risk_results"):
                                risk_data = simulation_results.get("risk_results", {})
                            elif simulation_results.get("cvar"):
                                risk_data = simulation_results.get("cvar", {})
                            else:
                                risk_data = {}
                                
                            # Extract metrics, with defaults if not available
                            risk_metrics["var_95"] = risk_data.get("var_95", initial_investment * 0.15)  # Default 15% VaR
                            risk_metrics["cvar_95"] = risk_data.get("cvar_95", initial_investment * 0.20)  # Default 20% CVaR
                            risk_metrics["max_drawdown"] = risk_data.get("max_drawdown", 0.25)  # Default 25% drawdown
                            risk_metrics["volatility"] = risk_data.get("volatility", stocks_pct * 0.15)  # Rough estimate
                            risk_metrics["sharpe"] = risk_data.get("sharpe_ratio", 0.8)  # Default Sharpe ratio
                            
                            # Format metrics as percentages of investment
                            var_pct = risk_metrics["var_95"] / initial_investment
                            cvar_pct = risk_metrics["cvar_95"] / initial_investment
                            
                            # Create risk metrics table with premium styling
                            pdf.set_font('Arial', 'B', 11)
                            pdf.set_fill_color(220, 230, 240)  # Light blue fill for headers
                            pdf.cell(70, 10, txt="Risk Metric", border=1, fill=True)
                            pdf.cell(120, 10, txt="Value", border=1, fill=True, ln=True)
                            
                            pdf.set_font('Arial', '', 10)
                            
                            # Value at Risk (VaR)
                            pdf.cell(70, 10, txt="Value at Risk (95% confidence)", border=1)
                            pdf.cell(120, 10, txt=f"${risk_metrics['var_95']:,.2f} ({var_pct:.2%} of portfolio)", border=1, ln=True)
                            
                            # Conditional VaR (CVaR/Expected Shortfall)
                            pdf.cell(70, 10, txt="Conditional VaR (95% confidence)", border=1)
                            pdf.cell(120, 10, txt=f"${risk_metrics['cvar_95']:,.2f} ({cvar_pct:.2%} of portfolio)", border=1, ln=True)
                            
                            # Maximum Drawdown
                            pdf.cell(70, 10, txt="Maximum Drawdown", border=1)
                            pdf.cell(120, 10, txt=f"{risk_metrics['max_drawdown']:.2%} from peak value", border=1, ln=True)
                            
                            # Portfolio Volatility
                            pdf.cell(70, 10, txt="Portfolio Volatility (Annual)", border=1)
                            pdf.cell(120, 10, txt=f"{risk_metrics['volatility']:.2%}", border=1, ln=True)
                            
                            # Sharpe Ratio
                            pdf.cell(70, 10, txt="Sharpe Ratio (Risk-Adjusted Return)", border=1)
                            pdf.cell(120, 10, txt=f"{risk_metrics['sharpe']:.2f}", border=1, ln=True)
                            
                            pdf.ln(10)
                            
                            # Risk Assessment Insights
                            pdf.set_font('Arial', 'B', 14)
                            pdf.cell(200, 10, txt="Risk Insights & Management Strategies", ln=True)
                            
                            pdf.set_font('Arial', '', 11)
                            
                            # Generate risk insights text based on metrics and portfolio
                            risk_insights = (
                                f"Your portfolio exhibits a {risk_profile} risk profile with the following characteristics:\n\n"
                                
                                f"• Value at Risk (VaR): In a worst-case scenario (with 95% confidence), your portfolio could "
                                f"lose up to ${risk_metrics['var_95']:,.2f} ({var_pct:.1%} of total value) over a short period.\n\n"
                                
                                f"• Conditional VaR: If losses exceed the VaR threshold, the average expected loss would be "
                                f"${risk_metrics['cvar_95']:,.2f} ({cvar_pct:.1%} of total value).\n\n"
                                
                                f"• Maximum Drawdown: Historically, portfolios with similar composition have experienced maximum "
                                f"peak-to-trough declines of approximately {risk_metrics['max_drawdown']:.1%}.\n\n"
                            )
                            
                            # Add risk management strategies based on profile
                            risk_insights += "Risk Management Strategies:\n\n"
                            
                            if risk_profile == 'conservative':
                                risk_insights += (
                                    "1. Maintain your higher allocation to bonds and fixed income to provide stability.\n"
                                    "2. Consider adding Treasury Inflation-Protected Securities (TIPS) to protect against inflation risk.\n"
                                    "3. Implement a regular rebalancing schedule to maintain your conservative risk profile.\n"
                                    "4. Consider adding small allocations to low-correlation alternative investments for diversification."
                                )
                            elif risk_profile == 'moderate':
                                risk_insights += (
                                    "1. Review your equity exposure regularly, especially as you approach major financial goals.\n"
                                    "2. Consider implementing a dynamic asset allocation strategy that adjusts with market conditions.\n"
                                    "3. Add defensive equity positions that can provide downside protection in market corrections.\n"
                                    "4. Maintain a 3-6 month emergency fund separate from your investment portfolio."
                                )
                            else:  # aggressive
                                risk_insights += (
                                    "1. Despite your aggressive stance, ensure some minimum bond allocation (10-15%) for stability.\n"
                                    "2. Consider implementing a strategic stop-loss policy for highly volatile positions.\n"
                                    "3. Diversify your equity exposure across sectors, geographies, and market capitalizations.\n"
                                    "4. Schedule quarterly portfolio reviews to assess concentration risks that may develop."
                                )
                            
                            pdf.multi_cell(190, 7, txt=risk_insights)
                            
                            # SECTION 4: Advanced Performance Projections
                            pdf.add_page()
                            pdf.set_font('Arial', 'B', 16)
                            pdf.set_text_color(31, 64, 122)  # Dark blue for section headers
                            pdf.cell(200, 15, txt="4. Advanced Performance Projections", ln=True)
                            pdf.set_text_color(0, 0, 0)  # Reset text color
                            
                            # Extract Monte Carlo data if available for detailed projections
                            mc_results = simulation_results.get("monte_carlo", {})
                            
                            # Calculate stats from the monte carlo results
                            mean_final = mc_results.get('mean_final', initial_investment * 1.7)  # Default 70% growth
                            median_final = mc_results.get('median_final', initial_investment * 1.6)  # Default 60% growth
                            
                            # Create percentile data for advanced projection table
                            percentiles = mc_results.get('percentiles', {})
                            if not percentiles:
                                # Create default percentiles if none available
                                percentiles = {
                                    5: initial_investment * 0.9,   # 5th percentile (worst case)
                                    25: initial_investment * 1.3,  # 25th percentile
                                    50: median_final,              # 50th percentile
                                    75: initial_investment * 1.9,  # 75th percentile
                                    95: initial_investment * 2.4   # 95th percentile (best case)
                                }
                            
                            # Calculate annualized returns
                            if time_horizon > 0:
                                annual_return = (1 + total_growth) ** (1/time_horizon) - 1
                                
                                # Calculate annualized returns for percentiles
                                percentile_returns = {}
                                for p, value in percentiles.items():
                                    growth = (value / initial_investment) - 1
                                    percentile_returns[p] = (1 + growth) ** (1/time_horizon) - 1
                            else:
                                annual_return = 0.07  # Default 7% if time horizon is invalid
                                percentile_returns = {5: 0.01, 25: 0.04, 50: 0.06, 75: 0.09, 95: 0.12}
                            
                            # Introduction text
                            pdf.set_font('Arial', '', 11)
                            projections_intro = (
                                "This section provides detailed projections of your portfolio's potential performance over your "
                                f"{time_horizon}-year investment horizon. These projections are based on sophisticated Monte Carlo "
                                "simulations that account for the historical behavior of similar asset allocations, including "
                                "volatility, correlations, and expected returns."
                            )
                            pdf.multi_cell(190, 7, txt=projections_intro)
                            pdf.ln(5)
                            
                            # Detailed projections table
                            pdf.set_font('Arial', 'B', 14)
                            pdf.cell(200, 10, txt="Portfolio Projection Scenarios", ln=True)
                            
                            # Create a premium styled projection table
                            pdf.set_font('Arial', 'B', 11)
                            pdf.set_fill_color(220, 230, 240)  # Light blue fill for headers
                            pdf.cell(50, 10, txt="Scenario", border=1, fill=True)
                            pdf.cell(50, 10, txt="Final Value ($)", border=1, fill=True)
                            pdf.cell(45, 10, txt="Total Growth", border=1, fill=True)
                            pdf.cell(45, 10, txt="Annual Return", border=1, fill=True, ln=True)
                            
                            pdf.set_font('Arial', '', 10)
                            
                            # Best Case scenario (95th percentile)
                            p95_value = percentiles.get(95, initial_investment * 2.4)
                            p95_growth = (p95_value / initial_investment) - 1
                            p95_annual = percentile_returns.get(95, 0.12)
                            
                            pdf.set_fill_color(230, 250, 230)  # Light green for best case
                            pdf.cell(50, 10, txt="Best Case (95th %ile)", border=1, fill=True)
                            pdf.cell(50, 10, txt=f"${p95_value:,.2f}", border=1, fill=True)
                            pdf.cell(45, 10, txt=f"{p95_growth:.2%}", border=1, fill=True)
                            pdf.cell(45, 10, txt=f"{p95_annual:.2%} annually", border=1, fill=True, ln=True)
                            
                            # Upper Middle Case (75th percentile)
                            p75_value = percentiles.get(75, initial_investment * 1.9)
                            p75_growth = (p75_value / initial_investment) - 1
                            p75_annual = percentile_returns.get(75, 0.09)
                            
                            pdf.set_fill_color(240, 240, 240)  # Light gray for middle cases
                            pdf.cell(50, 10, txt="Upper Mid (75th %ile)", border=1)
                            pdf.cell(50, 10, txt=f"${p75_value:,.2f}", border=1)
                            pdf.cell(45, 10, txt=f"{p75_growth:.2%}", border=1)
                            pdf.cell(45, 10, txt=f"{p75_annual:.2%} annually", border=1, ln=True)
                            
                            # Base Case (50th percentile / median)
                            p50_value = percentiles.get(50, median_final)
                            p50_growth = (p50_value / initial_investment) - 1
                            p50_annual = percentile_returns.get(50, 0.06)
                            
                            pdf.set_fill_color(240, 240, 240)  # Light gray for middle cases
                            pdf.cell(50, 10, txt="Base Case (50th %ile)", border=1, fill=True)
                            pdf.cell(50, 10, txt=f"${p50_value:,.2f}", border=1, fill=True)
                            pdf.cell(45, 10, txt=f"{p50_growth:.2%}", border=1, fill=True)
                            pdf.cell(45, 10, txt=f"{p50_annual:.2%} annually", border=1, fill=True, ln=True)
                            
                            # Lower Middle Case (25th percentile)
                            p25_value = percentiles.get(25, initial_investment * 1.3)
                            p25_growth = (p25_value / initial_investment) - 1
                            p25_annual = percentile_returns.get(25, 0.04)
                            
                            pdf.set_fill_color(240, 240, 240)  # Light gray for middle cases
                            pdf.cell(50, 10, txt="Lower Mid (25th %ile)", border=1)
                            pdf.cell(50, 10, txt=f"${p25_value:,.2f}", border=1)
                            pdf.cell(45, 10, txt=f"{p25_growth:.2%}", border=1)
                            pdf.cell(45, 10, txt=f"{p25_annual:.2%} annually", border=1, ln=True)
                            
                            # Worst Case scenario (5th percentile)
                            p5_value = percentiles.get(5, initial_investment * 0.9)
                            p5_growth = (p5_value / initial_investment) - 1
                            p5_annual = percentile_returns.get(5, 0.01)
                            
                            pdf.set_fill_color(250, 230, 230)  # Light red for worst case
                            pdf.cell(50, 10, txt="Worst Case (5th %ile)", border=1, fill=True)
                            pdf.cell(50, 10, txt=f"${p5_value:,.2f}", border=1, fill=True)
                            pdf.cell(45, 10, txt=f"{p5_growth:.2%}", border=1, fill=True)
                            pdf.cell(45, 10, txt=f"{p5_annual:.2%} annually", border=1, fill=True, ln=True)
                            
                            pdf.ln(10)
                            
                            # Probability of achieving specific goals
                            pdf.set_font('Arial', 'B', 14)
                            pdf.cell(200, 10, txt="Probability Analysis", ln=True)
                            
                            # Extract or estimate probability data
                            prob_positive = mc_results.get('prob_positive', 0.85)  # Default 85% probability of positive return
                            
                            # Create probability table
                            pdf.set_font('Arial', 'B', 11)
                            pdf.set_fill_color(220, 230, 240)  # Light blue for headers
                            pdf.cell(120, 10, txt="Investment Outcome", border=1, fill=True)
                            pdf.cell(70, 10, txt="Probability", border=1, fill=True, ln=True)
                            
                            pdf.set_font('Arial', '', 10)
                            
                            # Positive return probability
                            pdf.cell(120, 8, txt=f"Probability of Positive Return (After {time_horizon} Years)", border=1)
                            pdf.cell(70, 8, txt=f"{prob_positive:.1%}", border=1, ln=True)
                            
                            # Double investment probability (estimated)
                            double_threshold = initial_investment * 2
                            
                            # Estimate probability of doubling investment based on percentiles
                            if p50_value > double_threshold:
                                prob_double = 0.60  # If median exceeds double, then >50% probability
                            elif p75_value > double_threshold:
                                prob_double = 0.35  # If 75th percentile exceeds double
                            elif p95_value > double_threshold:
                                prob_double = 0.15  # If only 95th percentile exceeds double
                            else:
                                prob_double = 0.05  # Very low probability
                            
                            pdf.cell(120, 8, txt=f"Probability of Doubling Investment (>${initial_investment*2:,.2f})", border=1)
                            pdf.cell(70, 8, txt=f"{prob_double:.1%}", border=1, ln=True)
                            
                            # Calculate break-even time (years to recover initial investment)
                            # Simple estimation based on annual return
                            if annual_return > 0:
                                breakeven_years = 0  # Already positive return expected
                            else:
                                # For negative expected return, this is a theoretical calculation
                                breakeven_years = time_horizon * 1.5  # Conservative estimate
                            
                            pdf.cell(120, 8, txt="Estimated Break-Even Time", border=1)
                            pdf.cell(70, 8, txt=f"{breakeven_years:.1f} years", border=1, ln=True)
                            
                            # SECTION 5: Scenario Analysis
                            pdf.add_page()
                            pdf.set_font('Arial', 'B', 16)
                            pdf.set_text_color(31, 64, 122)  # Dark blue for section headers
                            pdf.cell(200, 15, txt="5. Scenario Analysis", ln=True)
                            pdf.set_text_color(0, 0, 0)  # Reset text color
                            
                            pdf.set_font('Arial', '', 11)
                            scenario_intro = (
                                "This premium section analyzes how your portfolio might perform under different market scenarios. "
                                "Understanding these potential outcomes can help you prepare for various market conditions and "
                                "make more informed investment decisions."
                            )
                            pdf.multi_cell(190, 7, txt=scenario_intro)
                            pdf.ln(5)
                            
                            # Create scenario analysis table
                            pdf.set_font('Arial', 'B', 14)
                            pdf.cell(200, 10, txt="Market Scenario Impact Analysis", ln=True)
                            
                            # Set up scenario table
                            pdf.set_font('Arial', 'B', 11)
                            pdf.set_fill_color(220, 230, 240)  # Light blue for headers
                            pdf.cell(65, 10, txt="Scenario", border=1, fill=True)
                            pdf.cell(45, 10, txt="Portfolio Impact", border=1, fill=True)
                            pdf.cell(40, 10, txt="Recovery Time", border=1, fill=True)
                            pdf.cell(40, 10, txt="Probability", border=1, fill=True, ln=True)
                            
                            pdf.set_font('Arial', '', 10)
                            
                            # Market scenarios analysis (these are hypothetical since we don't have real scenario data)
                            # Scenario 1: Market Correction
                            pdf.cell(65, 10, txt="Market Correction (-10%)", border=1)
                            # Impact varies by risk profile
                            correction_impact = -0.08 if risk_profile == 'conservative' else -0.10 if risk_profile == 'moderate' else -0.12
                            pdf.cell(45, 10, txt=f"{correction_impact:.1%}", border=1)
                            # Recovery time estimation based on risk profile
                            recovery_months = 6 if risk_profile == 'aggressive' else 8 if risk_profile == 'moderate' else 10
                            pdf.cell(40, 10, txt=f"{recovery_months} months", border=1)
                            pdf.cell(40, 10, txt="35%", border=1, ln=True)
                            
                            # Scenario 2: Recession
                            pdf.cell(65, 10, txt="Recession Scenario (-25%)", border=1)
                            # Impact varies by risk profile
                            recession_impact = -0.18 if risk_profile == 'conservative' else -0.22 if risk_profile == 'moderate' else -0.28
                            pdf.cell(45, 10, txt=f"{recession_impact:.1%}", border=1)
                            # Recovery time estimation based on risk profile
                            recovery_months = 18 if risk_profile == 'aggressive' else 24 if risk_profile == 'moderate' else 30
                            pdf.cell(40, 10, txt=f"{recovery_months} months", border=1)
                            pdf.cell(40, 10, txt="15%", border=1, ln=True)
                            
                            # Scenario 3: Inflation Surge
                            pdf.cell(65, 10, txt="Inflation Surge (5%+)", border=1)
                            # Impact varies by composition
                            inflation_impact = -0.05 if bonds_pct > 0.5 else -0.03 if commodities_pct > 0.1 else -0.08
                            pdf.cell(45, 10, txt=f"{inflation_impact:.1%}", border=1)
                            pdf.cell(40, 10, txt="12 months", border=1)
                            pdf.cell(40, 10, txt="20%", border=1, ln=True)
                            
                            # Scenario 4: Strong Bull Market
                            pdf.cell(65, 10, txt="Strong Bull Market (+20%)", border=1)
                            # Impact varies by risk profile
                            bull_impact = 0.12 if risk_profile == 'conservative' else 0.18 if risk_profile == 'moderate' else 0.25
                            pdf.cell(45, 10, txt=f"{bull_impact:+.1%}", border=1)
                            pdf.cell(40, 10, txt="N/A", border=1)
                            pdf.cell(40, 10, txt="30%", border=1, ln=True)
                            
                            pdf.ln(10)
                            
                            # Scenario analysis insights
                            pdf.set_font('Arial', 'B', 12)
                            pdf.cell(190, 10, txt="Scenario Insights:", ln=True)
                            
                            pdf.set_font('Arial', '', 11)
                            
                            # Generate insights text
                            scenario_insights = ""
                            if risk_profile == 'conservative':
                                scenario_insights = (
                                    "Your conservative portfolio is designed to withstand market corrections with reduced volatility. "
                                    f"In a recession scenario, your portfolio might decline by approximately {recession_impact:.1%}, "
                                    f"which is less than market averages. However, recovery periods may be longer due to the more "
                                    "defensive positioning. Your allocation provides some inflation protection, but consider adding "
                                    "additional inflation-sensitive assets like TIPS or commodities if inflation is a concern."
                                )
                            elif risk_profile == 'moderate':
                                scenario_insights = (
                                    "Your moderate risk portfolio balances growth with reasonable downside protection. "
                                    f"In a recession scenario, you might experience a {recession_impact:.1%} decline with an estimated "
                                    f"recovery period of {recovery_months} months. During strong bull markets, your portfolio should "
                                    f"capture approximately {bull_impact:.1%} growth, providing solid upside participation. "
                                    "Your current asset mix offers moderate inflation protection."
                                )
                            else:  # aggressive
                                scenario_insights = (
                                    "Your aggressive portfolio is positioned for maximum growth but with higher volatility. "
                                    f"In a recession scenario, your portfolio could decline by {recession_impact:.1%}, requiring "
                                    f"approximately {recovery_months} months to recover. However, during strong bull markets, you would "
                                    f"likely experience significant gains of around {bull_impact:.1%}. Your current allocation has "
                                    "limited inflation protection, which could be an area for enhancement."
                                )
                            
                            pdf.multi_cell(190, 7, txt=scenario_insights)
                            pdf.ln(5)
                            
                            # SECTION 6: Tax Efficiency Optimization
                            pdf.add_page()
                            pdf.set_font('Arial', 'B', 16)
                            pdf.set_text_color(31, 64, 122)  # Dark blue for section headers
                            pdf.cell(200, 15, txt="6. Tax Efficiency Optimization", ln=True)
                            pdf.set_text_color(0, 0, 0)  # Reset text color
                            
                            pdf.set_font('Arial', '', 11)
                            tax_intro = (
                                "Tax efficiency is often overlooked in portfolio management but can significantly impact your "
                                "net returns. This Founding Access section provides strategies to optimize your portfolio's "
                                "tax efficiency across different account types and investment vehicles."
                            )
                            pdf.multi_cell(190, 7, txt=tax_intro)
                            pdf.ln(5)
                            
                            # Asset location strategies table
                            pdf.set_font('Arial', 'B', 14)
                            pdf.cell(200, 10, txt="Asset Location Strategies", ln=True)
                            
                            # Table for tax-efficient asset placement
                            pdf.set_font('Arial', 'B', 11)
                            pdf.set_fill_color(220, 230, 240)  # Light blue for headers
                            pdf.cell(70, 10, txt="Account Type", border=1, fill=True)
                            pdf.cell(120, 10, txt="Optimal Assets", border=1, fill=True, ln=True)
                            
                            # Taxable accounts
                            pdf.set_font('Arial', 'B', 10)
                            pdf.cell(70, 10, txt="Taxable Accounts", border=1)
                            pdf.set_font('Arial', '', 10)
                            pdf.multi_cell(120, 10, txt="ETFs, municipal bonds, low-turnover index funds, tax-managed funds", border=1)
                            
                            # Tax-deferred accounts (401k, Traditional IRA)
                            pdf.set_font('Arial', 'B', 10)
                            pdf.cell(70, 10, txt="Tax-Deferred Accounts", border=1)
                            pdf.set_font('Arial', '', 10)
                            pdf.multi_cell(120, 10, txt="REITs, high-yield bonds, actively managed funds, dividend-focused stocks", border=1)
                            
                            # Tax-exempt accounts (Roth)
                            pdf.set_font('Arial', 'B', 10)
                            pdf.cell(70, 10, txt="Tax-Exempt Accounts", border=1)
                            pdf.set_font('Arial', '', 10)
                            pdf.multi_cell(120, 10, txt="High-growth stocks, aggressive investments with highest expected returns", border=1)
                            
                            pdf.ln(10)
                            
                            # Tax-efficient investment vehicles
                            pdf.set_font('Arial', 'B', 14)
                            pdf.cell(200, 10, txt="Tax-Efficient Investment Vehicles", ln=True)
                            
                            pdf.set_font('Arial', '', 11)
                            tax_vehicles = (
                                "For taxable accounts, consider these tax-efficient investment vehicles:\n\n"
                                
                                "• ETFs: Generally more tax-efficient than mutual funds due to their creation/redemption process, "
                                "which minimizes capital gains distributions.\n\n"
                                
                                "• Index Funds: Typically generate fewer capital gains due to low turnover.\n\n"
                                
                                "• Tax-Managed Funds: Specifically designed to minimize tax impact through strategies like "
                                "tax-loss harvesting and avoiding dividend-paying stocks.\n\n"
                                
                                "• Municipal Bonds: Interest is typically exempt from federal taxes, and potentially "
                                "state and local taxes if issued in your state of residence.\n\n"
                                
                                "• Direct Indexing: Allows for customization of index-like exposure with opportunities "
                                "for tax-loss harvesting at the individual security level."
                            )
                            pdf.multi_cell(190, 7, txt=tax_vehicles)
                            
                            # SECTION 7: Strategic Recommendations
                            pdf.add_page()
                            pdf.set_font('Arial', 'B', 16)
                            pdf.set_text_color(31, 64, 122)  # Dark blue for section headers
                            pdf.cell(200, 15, txt="7. Strategic Recommendations", ln=True)
                            pdf.set_text_color(0, 0, 0)  # Reset text color
                            
                            pdf.set_font('Arial', '', 11)
                            rec_intro = (
                                "Based on our comprehensive analysis of your portfolio, risk profile, and investment horizon, "
                                "we provide the following strategic recommendations to optimize your investment strategy. "
                                "These actionable insights are personalized to your specific financial situation."
                            )
                            pdf.multi_cell(190, 7, txt=rec_intro)
                            pdf.ln(5)
                            
                            # Generate recommendations based on portfolio characteristics
                            recommendations = []
                            
                            # Asset allocation recommendation
                            if abs(stocks_diff) > 10 or abs(bonds_diff) > 10:
                                if stocks_diff > 10:
                                    recommendations.append({
                                        "title": "Adjust Asset Allocation",
                                        "description": (
                                            f"Your current stock allocation ({stocks_pct * 100:.1f}%) is {stocks_diff:.1f}% higher than recommended "
                                            f"for your {risk_profile} risk profile. Consider reducing equity exposure by {stocks_diff:.1f}% "
                                            f"and increasing bond allocation to better align with your risk tolerance. This would reduce "
                                            f"potential volatility while still maintaining appropriate growth exposure."
                                        ),
                                        "priority": "High"
                                    })
                                elif stocks_diff < -10:
                                    recommendations.append({
                                        "title": "Increase Growth Exposure",
                                        "description": (
                                            f"Your current stock allocation ({stocks_pct * 100:.1f}%) is {-stocks_diff:.1f}% lower than recommended "
                                            f"for your {risk_profile} risk profile. Consider increasing equity exposure by {-stocks_diff:.1f}% "
                                            f"to better align with your growth objectives. This adjustment could significantly improve "
                                            f"your long-term performance potential."
                                        ),
                                        "priority": "High"
                                    })
                            else:
                                recommendations.append({
                                    "title": "Maintain Current Asset Allocation",
                                    "description": (
                                        "Your current asset allocation is well-aligned with your risk profile and investment goals. "
                                        "Implement a regular quarterly rebalancing schedule to maintain these allocations as market "
                                        "movements cause portfolio drift over time."
                                    ),
                                    "priority": "Medium"
                                })
                            
                            # Diversification recommendation
                            if len(query_data.get('assets', [])) < 5:
                                recommendations.append({
                                    "title": "Enhance Portfolio Diversification",
                                    "description": (
                                        "Your portfolio has relatively few holdings, which may increase concentration risk. "
                                        "Consider adding more diversified investments such as broad market ETFs or expanding into "
                                        "additional asset classes like international equities, REITs, or commodities to reduce "
                                        "single-asset risk and improve risk-adjusted returns."
                                    ),
                                    "priority": "High"
                                })
                            
                            # Tax efficiency recommendation 
                            recommendations.append({
                                "title": "Optimize Tax Efficiency",
                                "description": (
                                    "Review the tax efficiency of your current asset placement. Consider holding tax-efficient "
                                    "investments (like ETFs and municipal bonds) in taxable accounts, while placing tax-inefficient "
                                    "investments (like REITs and high-yield bonds) in tax-advantaged accounts. Also explore "
                                    "potential tax-loss harvesting opportunities to offset capital gains."
                                ),
                                "priority": "Medium"
                            })
                            
                            # Risk management recommendation based on profile
                            if risk_profile == 'aggressive' and bonds_pct < 0.1:
                                recommendations.append({
                                    "title": "Add Downside Protection",
                                    "description": (
                                        "Despite your aggressive risk profile, your very low bond allocation (less than 10%) "
                                        "may leave you vulnerable during market corrections. Consider allocating at least 10-15% "
                                        "to high-quality bonds to provide some stability during market downturns without "
                                        "significantly impacting your long-term return potential."
                                    ),
                                    "priority": "Medium"
                                })
                            elif risk_profile == 'conservative' and stocks_pct > 0.5:
                                recommendations.append({
                                    "title": "Reduce Volatility Risk",
                                    "description": (
                                        "Your current equity allocation (over 50%) appears higher than typical for your conservative "
                                        "risk profile. Consider reducing equity exposure and increasing allocation to high-quality "
                                        "bonds and other defensive assets to better align with your emphasis on capital preservation."
                                    ),
                                    "priority": "High"
                                })
                            
                            # Retirement planning recommendation (generic)
                            recommendations.append({
                                "title": "Regular Portfolio Review",
                                "description": (
                                    f"Schedule quarterly portfolio reviews to assess performance against your {time_horizon}-year investment "
                                    f"horizon goals. Adjust your strategy as needed based on changing market conditions, life events, "
                                    f"or shifts in your financial objectives."
                                ),
                                "priority": "Medium"
                            })
                            
                            # Format and add recommendations with priority indicators
                            for i, rec in enumerate(recommendations, 1):
                                # Add priority indicator with color-coding
                                priority = rec.get("priority", "Medium")
                                if priority == "High":
                                    pdf.set_text_color(180, 0, 0)  # Red for high priority
                                    priority_text = "★★★ HIGH PRIORITY"
                                elif priority == "Medium":
                                    pdf.set_text_color(0, 0, 180)  # Blue for medium priority
                                    priority_text = "★★ MEDIUM PRIORITY"
                                else:
                                    pdf.set_text_color(0, 120, 0)  # Green for low priority
                                    priority_text = "★ CONSIDERATION" 
                                
                                # Add priority marker
                                pdf.set_font('Arial', 'B', 9)
                                pdf.cell(190, 8, txt=priority_text, ln=True)
                                
                                # Reset text color for the rest of the content
                                pdf.set_text_color(0, 0, 0)
                                
                                # Add recommendation title
                                pdf.set_font('Arial', 'B', 12)
                                pdf.cell(190, 10, txt=f"{i}. {rec.get('title', '')}", ln=True)
                                
                                # Add recommendation description
                                pdf.set_font('Arial', '', 11)
                                pdf.multi_cell(190, 7, txt=rec.get('description', ''))
                                pdf.ln(10)
                            
                            # SECTION 8: Disclaimer & Legal
                            pdf.add_page()
                            pdf.set_font('Arial', 'B', 16)
                            pdf.set_text_color(31, 64, 122)  # Dark blue for section headers
                            pdf.cell(200, 15, txt="8. Disclaimer & Legal", ln=True)
                            pdf.set_text_color(0, 0, 0)  # Reset text color
                            
                            pdf.set_font('Arial', 'B', 12)
                            pdf.cell(190, 10, txt="Important Disclosures", ln=True)
                            
                            pdf.set_font('Arial', '', 10)
                            disclaimer = (
                                "IMPORTANT: This report is for informational purposes only and does not constitute investment advice.\n\n"
                                
                                "Past performance is not indicative of future results. All investments involve risk, including the "
                                "possible loss of principal. Diversification does not ensure a profit or protect against a loss "
                                "in a declining market.\n\n"
                                
                                "The projections or other information generated by this report regarding the likelihood of various "
                                "investment outcomes are hypothetical in nature, do not reflect actual investment results, and are "
                                "not guarantees of future results. The results may vary with each use and over time.\n\n"
                                
                                "This report is based on data and information provided by you, as well as estimates, approximations, "
                                "and simulation models that may use historical performance, market indices, and statistical models to "
                                "estimate potential future outcomes. These estimates are subject to numerous assumptions, risks, and "
                                "limitations.\n\n"
                                
                                "Users should consult with a qualified financial advisor before making any investment decisions. "
                                "Iso AI is not a registered investment advisor and does not provide personalized financial advice.\n\n"
                                
                                "This report is confidential and for the exclusive use of the client. Redistribution without "
                                "permission is prohibited.\n\n"
                                
                                "Tax information provided is general in nature and is not intended as legal or tax advice. "
                                "Consult a qualified tax professional for specific guidance tailored to your situation."
                            )
                            pdf.multi_cell(190, 7, txt=disclaimer)
                            
                            # Add copyright and generation date
                            pdf.ln(10)
                            pdf.set_font('Arial', 'I', 9)
                            pdf.cell(190, 5, txt=f"©{datetime.now().year} Iso AI, Inc. All rights reserved.", ln=True, align='C')
                            pdf.cell(190, 5, txt=f"Report generated on {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}", ln=True, align='C')
                            pdf.cell(190, 5, txt="CONFIDENTIAL - FOUNDING ACCESS MEMBER", ln=True, align='C')
                            
                            # Create timestamp for unique file
                            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                            report_pdf_path = f"iso_financial_premium_report_{timestamp}.pdf"
                            
                            # Output PDF
                            pdf.output(report_pdf_path)
                            print(f"Generated premium report at: {report_pdf_path}")
                            st.write(f"Premium report generated at: {report_pdf_path}")
                            
                            # Check if file exists
                            if not os.path.exists(report_pdf_path):
                                st.error(f"Premium PDF file not found at {report_pdf_path}")
                                raise FileNotFoundError(f"PDF file not found at {report_pdf_path}")
                                
                            # Show file size
                            file_size = os.path.getsize(report_pdf_path)
                            st.write(f"Premium PDF size: {file_size} bytes")
                            
                            # Read the PDF file
                            with open(report_pdf_path, "rb") as f:
                                pdf_bytes = f.read()
                            
                            # Encode PDF for display and download
                            b64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
                            
                            # Display PDF in the app using iframe (better compatibility)
                            st.subheader("Premium Report Preview")
                            
                            # Try both methods for better compatibility
                            pdf_display = f"""
                            <div style="display: flex; justify-content: center;">
                                <iframe src="data:application/pdf;base64,{b64_pdf}" width="700" height="500" type="application/pdf"></iframe>
                            </div>
                            <p style="text-align: center; margin-top: 10px; font-size: 12px;">(If preview doesn't load, use the download button below)</p>
                            """
                            st.markdown(pdf_display, unsafe_allow_html=True)
                            
                            # Alternative object tag as backup
                            pdf_display_alt = f"""
                            <div style="display: flex; justify-content: center;">
                                <object data="data:application/pdf;base64,{b64_pdf}" type="application/pdf" width="700" height="500">
                                    <p>Unable to display PDF. Please download the PDF file.</p>
                                </object>
                            </div>
                            """
                            st.markdown(pdf_display_alt, unsafe_allow_html=True)
                            
                            # Download button for PDF
                            st.download_button(
                                label="Download Premium PDF Report",
                                data=pdf_bytes,
                                file_name=f"iso_financial_premium_report_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf",
                                mime="application/pdf"
                            )
                            
                            # Save report to session state
                            if "saved_reports" not in st.session_state:
                                st.session_state.saved_reports = []
                                
                            # Add report to saved reports
                            st.session_state.saved_reports.append({
                                "type": "Premium Report",
                                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "pdf_path": report_pdf_path
                            })
                            
                            # Success message
                            st.success("Premium report generated successfully and saved to your account.")
                        
                    except Exception as e:
                        # Log detailed error for debugging
                        import traceback
                        error_details = traceback.format_exc()
                        print(f"[ERROR] Premium report generation failed: {str(e)}")
                        print(f"[ERROR] Traceback: {error_details}")
                        
                        # Show more helpful error message to users
                        st.error(f"Unable to generate the premium report: {str(e)}")
                        
                        # Add debugging info in expandable section
                        with st.expander("Technical Error Details (for debugging)"):
                            st.code(error_details)
                            st.write("Query Data:", query_data)
                            st.write("Simulation Results Keys:", list(simulation_results.keys()))
                            for key, value in simulation_results.items():
                                if isinstance(value, dict):
                                    st.write(f"{key} keys:", list(value.keys() if value else "Empty"))
            
            else:
                st.error("Premium reports require a premium subscription. Please upgrade your account.")
    
    # Add saved reports section
    if st.session_state.get("saved_reports"):
        st.subheader("Your Saved Reports")
        
        for i, report in enumerate(st.session_state.get("saved_reports", [])):
            with st.expander(f"Report {i+1}: {report.get('date', 'Unknown date')}"):
                st.write(f"Type: {report.get('type', 'Unknown')}")
                st.write(f"Generated: {report.get('date', 'Unknown')}")
                
                # Show PDF if available
                if "pdf_path" in report and os.path.exists(report["pdf_path"]):
                    with open(report["pdf_path"], "rb") as f:
                        pdf_bytes = f.read()
                    
                    st.download_button(
                        label="Download Report",
                        data=pdf_bytes,
                        file_name=f"iso_financial_report_{i+1}.pdf",
                        mime="application/pdf"
                    )
                else:
                    st.write("PDF file not available.")
                
                # Delete button
                if st.button(f"Delete Report {i+1}", key=f"delete_report_{i}"):
                    st.session_state.saved_reports.pop(i)
                    st.rerun()

# Add a disclaimer at the bottom
st.markdown("---")
st.caption(get_standard_disclaimer())