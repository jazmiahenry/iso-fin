import os
import tempfile
import datetime
import logging
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fpdf import FPDF
import qrcode
import json

# Setup logger
logger = logging.getLogger(__name__)

class FreeReportGenerator:
    """Generator for free financial reports with comprehensive analysis"""
    
    def __init__(self):
        """Initialize the free report generator"""
        self.temp_files = []
        self.colors = {
            'header_bg': (230, 230, 250),  # Light lavender for headers
            'subheader_bg': (240, 240, 245),  # Lighter background for subheaders
            'highlight_bg': (235, 245, 250),  # Light blue highlight
            'positive': (0, 128, 0),  # Green for positive values
            'negative': (220, 20, 60),  # Crimson for negative values
            'neutral': (100, 100, 100),  # Gray for neutral text
        }
        print("FreeReportGenerator initialized")
    
    def generate_report(self, **kwargs):
        """
        Generate a comprehensive free report with the provided simulation data.
        
        Args:
            portfolio: Dictionary mapping ticker symbols to weights
            asset_metadata: Dictionary with metadata for each asset (sector, asset_class, etc.)
            monte_carlo_results: Results from Monte Carlo simulation
            scenario_results: Results from scenario analysis
            risk_results: Results from risk analysis
            optimization_results: Results from portfolio optimization
            market_data: Dictionary with market data for each asset
            sections: Dictionary specifying which sections to include
            output_file: Optional output file path
            
        Returns:
            Path to the generated PDF file
        """
        print(f"generate_report called with kwargs: {kwargs}")
        
        # Extract parameters
        portfolio = kwargs.get('portfolio', {})
        asset_metadata = kwargs.get('asset_metadata', {})
        monte_carlo_results = kwargs.get('monte_carlo_results', {})
        scenario_results = kwargs.get('scenario_results', {})
        risk_results = kwargs.get('risk_results', {})
        optimization_results = kwargs.get('optimization_results', {})
        market_data = kwargs.get('market_data', {})
        sections = kwargs.get('sections', {})
        output_file = kwargs.get('output_file')
        
        # Create a PDF report
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        if output_file is None:
            output_file = f"iso_financial_free_report_{timestamp}.pdf"
        
        # Generate a comprehensive PDF report
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # Title page
        pdf.add_page()
        pdf.set_font("Arial", 'B', size=20)
        pdf.cell(200, 20, txt="ISO Financial Portfolio Analysis", ln=True, align='C')
        pdf.set_font("Arial", 'B', size=16)
        pdf.cell(200, 10, txt="Free Report", ln=True, align='C')
        
        # Add date and time
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')
        
        # Table of contents
        pdf.ln(20)
        pdf.set_font("Arial", 'B', size=14)
        pdf.cell(200, 10, txt="Report Contents", ln=True)
        
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 8, txt="1. Portfolio Summary", ln=True)
        pdf.cell(200, 8, txt="2. Asset Allocation Analysis", ln=True)
        pdf.cell(200, 8, txt="3. Risk Assessment", ln=True)
        pdf.cell(200, 8, txt="4. Performance Projections", ln=True)
        pdf.cell(200, 8, txt="5. Recommendations", ln=True)
        pdf.cell(200, 8, txt="6. Disclaimer & Legal Information", ln=True)
        
        #-----------------------
        # 1. Portfolio Summary
        #-----------------------
        pdf.add_page()
        # Section header with background
        pdf.set_fill_color(*self.colors['header_bg'])
        pdf.set_font("Arial", 'B', size=16)
        pdf.cell(190, 10, txt="1. Portfolio Summary", ln=True, fill=True)
        pdf.ln(5)
        
        # Portfolio composition
        pdf.set_font("Arial", 'B', size=14)
        pdf.cell(190, 10, txt="Portfolio Overview", ln=True)
        
        # Get the initial investment amount
        initial_investment = monte_carlo_results.get('initial_investment', 100000)
        if hasattr(initial_investment, 'item'):  # Handle numpy types
            initial_investment = initial_investment.item()
        
        # Key metrics
        pdf.set_font("Arial", size=12)
        pdf.cell(100, 10, txt=f"Total Assets: {len(portfolio)}", ln=False)
        pdf.cell(90, 10, txt=f"Initial Investment: ${initial_investment:,.2f}", ln=True)
        
        # Create a formatted table for portfolio assets
        pdf.ln(5)
        pdf.set_font("Arial", 'B', size=12)
        
        # Table header with colored background
        pdf.set_fill_color(*self.colors['subheader_bg'])
        pdf.cell(45, 10, txt="Asset", border=1, fill=True)
        pdf.cell(35, 10, txt="Weight", border=1, fill=True)
        pdf.cell(55, 10, txt="Amount ($)", border=1, fill=True)
        pdf.cell(55, 10, txt="Asset Class", border=1, fill=True, ln=True)
        
        # Sort assets by weight descending
        sorted_portfolio = sorted(portfolio.items(), key=lambda x: x[1], reverse=True)
        
        # Table rows
        pdf.set_font("Arial", size=11)
        for ticker, weight in sorted_portfolio:
            # Calculate dollar amount
            amount = initial_investment * weight
            
            # Get asset class if available, or default to "Equity"
            asset_class = "Equity"  # Default
            if ticker in asset_metadata:
                asset_class = asset_metadata.get(ticker, {}).get('asset_class', "Equity")
            
            pdf.cell(45, 8, txt=f"{ticker}", border=1)
            pdf.cell(35, 8, txt=f"{weight:.2%}", border=1)
            pdf.cell(55, 8, txt=f"${amount:,.2f}", border=1)
            pdf.cell(55, 8, txt=f"{asset_class}", border=1, ln=True)
        
        # Summary of key financial metrics
        pdf.ln(10)
        pdf.set_font("Arial", 'B', size=14)
        pdf.cell(190, 10, txt="Key Portfolio Metrics", ln=True)
        
        # Extract expected return and volatility
        expected_return = monte_carlo_results.get('expected_return', 0.08)  # Default to 8%
        portfolio_volatility = risk_results.get('portfolio_volatility', 0.15)  # Default to 15%
        sharpe_ratio = risk_results.get('sharpe_ratio', 0.5)  # Default to 0.5
        
        # Format a metrics table with colored background
        pdf.ln(5)
        pdf.set_fill_color(*self.colors['highlight_bg'])
        pdf.set_font("Arial", 'B', size=11)
        pdf.cell(63, 10, txt="Expected Annual Return", border=1, fill=True)
        pdf.cell(63, 10, txt="Annual Volatility", border=1, fill=True)
        pdf.cell(64, 10, txt="Sharpe Ratio", border=1, fill=True, ln=True)
        
        pdf.set_font("Arial", size=11)
        pdf.cell(63, 10, txt=f"{expected_return:.2%}", border=1)
        pdf.cell(63, 10, txt=f"{portfolio_volatility:.2%}", border=1)
        pdf.cell(64, 10, txt=f"{sharpe_ratio:.2f}", border=1, ln=True)
        
        #-----------------------
        # 2. Asset Allocation Analysis
        #-----------------------
        pdf.add_page()
        pdf.set_fill_color(*self.colors['header_bg'])
        pdf.set_font("Arial", 'B', size=16)
        pdf.cell(190, 10, txt="2. Asset Allocation Analysis", ln=True, fill=True)
        pdf.ln(5)
        
        # Categorize assets by type
        asset_types = {}
        for ticker, weight in portfolio.items():
            asset_type = asset_metadata.get(ticker, {}).get('asset_class', "Equity")
            asset_types[asset_type] = asset_types.get(asset_type, 0) + weight
        
        # Asset Allocation by Type
        pdf.set_font("Arial", 'B', size=14)
        pdf.cell(190, 10, txt="Allocation by Asset Class", ln=True)
        
        # Create a table for asset allocation
        pdf.ln(5)
        pdf.set_fill_color(*self.colors['subheader_bg'])
        pdf.set_font("Arial", 'B', size=11)
        pdf.cell(95, 10, txt="Asset Class", border=1, fill=True)
        pdf.cell(45, 10, txt="Allocation", border=1, fill=True)
        pdf.cell(50, 10, txt="Amount ($)", border=1, fill=True, ln=True)
        
        # Table rows sorted by allocation percentage (descending)
        pdf.set_font("Arial", size=11)
        for asset_type, weight in sorted(asset_types.items(), key=lambda x: x[1], reverse=True):
            amount = initial_investment * weight
            pdf.cell(95, 8, txt=f"{asset_type}", border=1)
            pdf.cell(45, 8, txt=f"{weight:.2%}", border=1)
            pdf.cell(50, 8, txt=f"${amount:,.2f}", border=1, ln=True)
        
        # Sector Allocation (if available)
        pdf.ln(10)
        pdf.set_font("Arial", 'B', size=14)
        pdf.cell(190, 10, txt="Allocation by Sector", ln=True)
        
        # Categorize assets by sector
        sectors = {}
        has_sector_data = False
        
        for ticker, weight in portfolio.items():
            sector = asset_metadata.get(ticker, {}).get('sector', None)
            if sector:
                has_sector_data = True
                sectors[sector] = sectors.get(sector, 0) + weight
        
        if has_sector_data:
            # Create a table for sector allocation
            pdf.ln(5)
            pdf.set_fill_color(*self.colors['subheader_bg'])
            pdf.set_font("Arial", 'B', size=11)
            pdf.cell(95, 10, txt="Sector", border=1, fill=True)
            pdf.cell(45, 10, txt="Allocation", border=1, fill=True)
            pdf.cell(50, 10, txt="Amount ($)", border=1, fill=True, ln=True)
            
            # Table rows sorted by allocation percentage (descending)
            pdf.set_font("Arial", size=11)
            for sector, weight in sorted(sectors.items(), key=lambda x: x[1], reverse=True):
                amount = initial_investment * weight
                pdf.cell(95, 8, txt=f"{sector}", border=1)
                pdf.cell(45, 8, txt=f"{weight:.2%}", border=1)
                pdf.cell(50, 8, txt=f"${amount:,.2f}", border=1, ln=True)
        else:
            pdf.set_font("Arial", 'I', size=11)
            pdf.cell(190, 10, txt="Sector data not available for this portfolio.", ln=True)
        
        # Diversification Analysis
        pdf.ln(10)
        pdf.set_font("Arial", 'B', size=14)
        pdf.cell(190, 10, txt="Diversification Analysis", ln=True)
        
        # Calculate diversification metrics
        num_assets = len(portfolio)
        top_holdings_weight = sum(sorted(portfolio.values(), reverse=True)[:min(3, num_assets)])
        diversification_score = 1 - (top_holdings_weight / min(1, sum(portfolio.values())))
        
        # Create a diversification summary
        pdf.set_font("Arial", size=11)
        pdf.multi_cell(190, 6, txt="Diversification helps reduce risk by spreading investments across multiple assets. " +
                                  "A well-diversified portfolio typically includes a mix of asset classes and sectors.")
        
        pdf.ln(5)
        pdf.set_fill_color(*self.colors['highlight_bg'])
        pdf.set_font("Arial", 'B', size=11)
        pdf.cell(63, 10, txt="Number of Assets", border=1, fill=True)
        pdf.cell(63, 10, txt="Top 3 Holdings Weight", border=1, fill=True)
        pdf.cell(64, 10, txt="Diversification Score", border=1, fill=True, ln=True)
        
        pdf.set_font("Arial", size=11)
        pdf.cell(63, 10, txt=f"{num_assets}", border=1)
        pdf.cell(63, 10, txt=f"{top_holdings_weight:.2%}", border=1)
        
        # Color code the diversification score
        if diversification_score >= 0.7:
            text_color = self.colors['positive']
            score_text = "Excellent"
        elif diversification_score >= 0.5:
            text_color = (0, 128, 128)  # Teal - moderately good
            score_text = "Good"
        else:
            text_color = self.colors['negative']
            score_text = "Needs Improvement"
        
        pdf.set_text_color(*text_color)
        pdf.cell(64, 10, txt=f"{diversification_score:.2f} ({score_text})", border=1, ln=True)
        pdf.set_text_color(0, 0, 0)  # Reset text color to black
        
        #-----------------------
        # 3. Risk Assessment
        #-----------------------
        pdf.add_page()
        pdf.set_fill_color(*self.colors['header_bg'])
        pdf.set_font("Arial", 'B', size=16)
        pdf.cell(190, 10, txt="3. Risk Assessment", ln=True, fill=True)
        pdf.ln(5)
        
        # Risk overview
        pdf.set_font("Arial", size=11)
        pdf.multi_cell(190, 6, txt="This section analyzes the risk characteristics of your portfolio. " +
                                  "Understanding risk helps you make informed investment decisions " +
                                  "and prepare for potential market downturns.")
        
        # Core risk metrics
        pdf.ln(5)
        pdf.set_font("Arial", 'B', size=14)
        pdf.cell(190, 10, txt="Key Risk Metrics", ln=True)
        
        # Extract risk metrics
        var_95 = risk_results.get('var_95', initial_investment * 0.1)  # Default to 10% of initial
        cvar_95 = risk_results.get('cvar_95', initial_investment * 0.15)  # Default to 15% of initial
        var_99 = risk_results.get('var_99', initial_investment * 0.15)  # Default to 15% of initial
        max_drawdown = risk_results.get('max_drawdown', 0.25)  # Default to 25%
        beta = risk_results.get('beta', 1.0)  # Default to 1.0
        
        # Create a table for risk metrics with colored background
        pdf.ln(5)
        pdf.set_fill_color(*self.colors['subheader_bg'])
        pdf.set_font("Arial", 'B', size=11)
        pdf.cell(47, 10, txt="Value at Risk (95%)", border=1, fill=True)
        pdf.cell(47, 10, txt="Conditional VaR", border=1, fill=True)
        pdf.cell(48, 10, txt="Maximum Drawdown", border=1, fill=True)
        pdf.cell(48, 10, txt="Portfolio Beta", border=1, fill=True, ln=True)
        
        pdf.set_font("Arial", size=11)
        pdf.cell(47, 10, txt=f"${var_95:,.2f}", border=1)
        pdf.cell(47, 10, txt=f"${cvar_95:,.2f}", border=1)
        pdf.cell(48, 10, txt=f"{max_drawdown:.2%}", border=1)
        pdf.cell(48, 10, txt=f"{beta:.2f}", border=1, ln=True)
        
        # Risk explanations
        pdf.ln(8)
        pdf.set_font("Arial", 'B', size=12)
        pdf.cell(190, 8, txt="Understanding These Risk Measures:", ln=True)
        
        pdf.set_font("Arial", size=11)
        pdf.multi_cell(190, 6, txt="• Value at Risk (VaR): The maximum loss expected over a specified time period at a 95% confidence level. " +
                                  "This means there is a 5% chance your portfolio could lose more than this amount.")
        
        pdf.multi_cell(190, 6, txt="• Conditional VaR (CVaR): The expected loss if the worst-case threshold (VaR) is exceeded. " +
                                  "This gives insight into the severity of potential extreme losses.")
        
        pdf.multi_cell(190, 6, txt="• Maximum Drawdown: The largest percentage drop from peak to trough in portfolio value, " +
                                  "based on historical data or simulation. This helps you understand potential declines.")
        
        pdf.multi_cell(190, 6, txt="• Portfolio Beta: A measure of your portfolio's volatility compared to the overall market. " +
                                  "A beta of 1 indicates movement similar to the market; higher values indicate more volatility.")
        
        # Risk profile assessment
        pdf.ln(8)
        pdf.set_font("Arial", 'B', size=14)
        pdf.cell(190, 10, txt="Portfolio Risk Profile", ln=True)
        
        # Determine risk profile based on volatility and beta
        risk_profile = "Moderate"
        if portfolio_volatility <= 0.1 and beta <= 0.8:
            risk_profile = "Conservative"
        elif portfolio_volatility >= 0.18 or beta >= 1.2:
            risk_profile = "Aggressive"
        
        # Risk profile explanation based on the determined profile
        risk_descriptions = {
            "Conservative": 
                "Your portfolio has a conservative risk profile, characterized by lower volatility and potentially lower returns. " +
                "It's designed to preserve capital while providing modest growth, suitable for investors with lower risk tolerance " +
                "or shorter time horizons.",
            
            "Moderate": 
                "Your portfolio has a moderate risk profile, balancing growth potential with risk mitigation. " +
                "This balanced approach aims to capture market upside while providing some downside protection, " +
                "suitable for investors with medium-term horizons and average risk tolerance.",
            
            "Aggressive": 
                "Your portfolio has an aggressive risk profile, focused on maximizing growth potential with higher volatility. " +
                "This approach seeks higher returns by accepting more short-term fluctuations, suitable for investors " +
                "with longer time horizons and higher risk tolerance."
        }
        
        # Display risk profile with colored background
        pdf.ln(5)
        pdf.set_fill_color(*self.colors['highlight_bg'])
        pdf.set_font("Arial", 'B', size=12)
        pdf.cell(190, 10, txt=f"Risk Profile Assessment: {risk_profile}", ln=True, fill=True)
        
        pdf.set_font("Arial", size=11)
        pdf.multi_cell(190, 6, txt=risk_descriptions.get(risk_profile, ""))
        
        #-----------------------
        # 4. Performance Projections
        #-----------------------
        pdf.add_page()
        pdf.set_fill_color(*self.colors['header_bg'])
        pdf.set_font("Arial", 'B', size=16)
        pdf.cell(190, 10, txt="4. Performance Projections", ln=True, fill=True)
        pdf.ln(5)
        
        # Performance projections overview
        pdf.set_font("Arial", size=11)
        pdf.multi_cell(190, 6, txt="This section projects the potential future performance of your portfolio " +
                                  "based on Monte Carlo simulations. These projections help visualize possible " +
                                  "outcomes over different time horizons.")
        
        # Monte Carlo Results summary
        if monte_carlo_results:
            pdf.ln(5)
            pdf.set_font("Arial", 'B', size=14)
            pdf.cell(190, 10, txt="Simulation Results", ln=True)
            
            # Get simulation parameters and results
            mean_final = monte_carlo_results.get('mean_final', initial_investment * 1.5)
            median_final = monte_carlo_results.get('median_final', initial_investment * 1.4)
            prob_positive = monte_carlo_results.get('prob_positive', 0.75)
            time_horizon = monte_carlo_results.get('time_horizon', 10)  # years
            
            # Calculate growth metrics
            growth_pct = (mean_final - initial_investment) / initial_investment * 100
            annualized_return = ((mean_final / initial_investment) ** (1 / time_horizon)) - 1
            
            # Create formatted summary boxes
            pdf.ln(5)
            pdf.set_fill_color(*self.colors['highlight_bg'])
            pdf.set_font("Arial", 'B', size=11)
            pdf.cell(95, 10, txt="Initial Investment", border=1, fill=True)
            pdf.cell(95, 10, txt="Projected Final Value (Mean)", border=1, fill=True, ln=True)
            
            pdf.set_font("Arial", size=11)
            pdf.cell(95, 10, txt=f"${initial_investment:,.2f}", border=1)
            pdf.cell(95, 10, txt=f"${mean_final:,.2f}", border=1, ln=True)
            
            pdf.set_font("Arial", 'B', size=11)
            pdf.cell(63, 10, txt="Total Growth", border=1, fill=True)
            pdf.cell(63, 10, txt="Annualized Return", border=1, fill=True)
            pdf.cell(64, 10, txt="Success Probability", border=1, fill=True, ln=True)
            
            pdf.set_font("Arial", size=11)
            pdf.cell(63, 10, txt=f"{growth_pct:.2f}%", border=1)
            pdf.cell(63, 10, txt=f"{annualized_return:.2%}", border=1)
            pdf.cell(64, 10, txt=f"{prob_positive:.1%}", border=1, ln=True)
            
            # Percentile distribution
            percentiles = monte_carlo_results.get('percentiles', {})
            if percentiles:
                pdf.ln(10)
                pdf.set_font("Arial", 'B', size=14)
                pdf.cell(190, 10, txt=f"Projected Outcomes ({time_horizon}-Year Horizon)", ln=True)
                
                pdf.ln(5)
                pdf.set_fill_color(*self.colors['subheader_bg'])
                pdf.set_font("Arial", 'B', size=11)
                pdf.cell(47, 10, txt="Scenario", border=1, fill=True)
                pdf.cell(48, 10, txt="Final Value", border=1, fill=True)
                pdf.cell(47, 10, txt="Total Return", border=1, fill=True)
                pdf.cell(48, 10, txt="Annualized", border=1, fill=True, ln=True)
                
                # Sort percentiles and display them
                scenarios = {
                    10: "Pessimistic",
                    25: "Below Average",
                    50: "Average (Median)",
                    75: "Above Average",
                    90: "Optimistic"
                }
                
                pdf.set_font("Arial", size=11)
                for pct in sorted(percentiles.keys()):
                    if pct in scenarios:
                        value = percentiles[pct]
                        total_return = ((value - initial_investment) / initial_investment)
                        ann_return = ((value / initial_investment) ** (1 / time_horizon)) - 1
                        
                        scenario_text = scenarios[pct]
                        
                        # Adjust text color based on scenario
                        if pct <= 25:
                            pdf.set_text_color(*self.colors['negative'])
                        elif pct >= 75:
                            pdf.set_text_color(*self.colors['positive'])
                        else:
                            pdf.set_text_color(*self.colors['neutral'])
                        
                        pdf.cell(47, 10, txt=f"{scenario_text}", border=1)
                        pdf.cell(48, 10, txt=f"${value:,.2f}", border=1)
                        pdf.cell(47, 10, txt=f"{total_return:.2%}", border=1)
                        pdf.cell(48, 10, txt=f"{ann_return:.2%}", border=1, ln=True)
                
                pdf.set_text_color(0, 0, 0)  # Reset text color to black
                
                # Interpretation of results
                pdf.ln(8)
                pdf.set_font("Arial", 'B', size=12)
                pdf.cell(190, 8, txt="Interpretation:", ln=True)
                
                pdf.set_font("Arial", size=11)
                pdf.multi_cell(190, 6, txt=f"• There is a {prob_positive:.1%} probability that your portfolio will grow over the {time_horizon}-year period.")
                pdf.multi_cell(190, 6, txt=f"• The median projected outcome is ${median_final:,.2f}, representing a {((median_final/initial_investment)-1):.2%} total return.")
                
                # Investment growth timeline (simplified projection)
                pdf.ln(8)
                pdf.set_font("Arial", 'B', size=14)
                pdf.cell(190, 10, txt="Projected Growth Timeline", ln=True)
                
                # Create a simplified growth timeline table
                pdf.ln(5)
                pdf.set_fill_color(*self.colors['subheader_bg'])
                pdf.set_font("Arial", 'B', size=11)
                pdf.cell(38, 10, txt="Year", border=1, fill=True)
                pdf.cell(76, 10, txt="Projected Value (Median)", border=1, fill=True)
                pdf.cell(76, 10, txt="Cumulative Return", border=1, fill=True, ln=True)
                
                # Generate simplified growth timeline
                timeline_years = [0, time_horizon // 4, time_horizon // 2, 3 * time_horizon // 4, time_horizon]
                timeline_years = sorted(list(set(timeline_years)))  # Remove duplicates
                
                pdf.set_font("Arial", size=11)
                for year in timeline_years:
                    if year == 0:
                        value = initial_investment
                        return_pct = 0
                    else:
                        # Simple compound growth formula based on annualized return
                        value = initial_investment * (1 + annualized_return) ** year
                        return_pct = (value - initial_investment) / initial_investment
                    
                    pdf.cell(38, 10, txt=f"Year {year}", border=1)
                    pdf.cell(76, 10, txt=f"${value:,.2f}", border=1)
                    pdf.cell(76, 10, txt=f"{return_pct:.2%}", border=1, ln=True)
        
        #-----------------------
        # 5. Recommendations
        #-----------------------
        pdf.add_page()
        pdf.set_fill_color(*self.colors['header_bg'])
        pdf.set_font("Arial", 'B', size=16)
        pdf.cell(190, 10, txt="5. Recommendations", ln=True, fill=True)
        pdf.ln(5)
        
        # Introduction to recommendations
        pdf.set_font("Arial", size=11)
        pdf.multi_cell(190, 6, txt="Based on the analysis of your portfolio, we provide the following recommendations " +
                                  "to help optimize your investment strategy. These suggestions are tailored to your " +
                                  "current asset allocation and risk profile.")
        
        # Diversification recommendations
        pdf.ln(8)
        pdf.set_font("Arial", 'B', size=14)
        pdf.cell(190, 10, txt="Diversification Recommendations", ln=True)
        
        # Generate diversification recommendations based on portfolio characteristics
        pdf.set_font("Arial", size=11)
        
        # Check asset class diversification
        if len(asset_types) < 3:
            pdf.ln(2)
            pdf.set_fill_color(*self.colors['highlight_bg'])
            pdf.cell(10, 10, txt="➤", ln=False, fill=False)
            pdf.multi_cell(180, 10, txt="Consider adding more asset classes to your portfolio for better diversification.", fill=False)
            
            # Suggest asset classes that are missing
            missing_asset_classes = []
            common_asset_classes = ["Equity", "Fixed Income", "Real Estate", "Commodities", "Cash"]
            for asset_class in common_asset_classes:
                if asset_class not in asset_types:
                    missing_asset_classes.append(asset_class)
            
            if missing_asset_classes:
                missing_classes_text = ", ".join(missing_asset_classes)
                pdf.multi_cell(180, 6, txt=f"    Consider adding exposure to: {missing_classes_text}")
        
        # Check sector diversification for equity-heavy portfolios
        if has_sector_data and 'Equity' in asset_types and asset_types.get('Equity', 0) > 0.6:
            # Check if any sector has more than 25% allocation
            overweight_sectors = []
            for sector, weight in sectors.items():
                sector_rel_weight = weight / asset_types.get('Equity', 1)
                if sector_rel_weight > 0.25:
                    overweight_sectors.append((sector, sector_rel_weight))
            
            if overweight_sectors:
                pdf.ln(5)
                pdf.cell(10, 10, txt="➤", ln=False)
                pdf.multi_cell(180, 10, txt="Consider reducing concentration in the following sectors:")
                
                for sector, weight in overweight_sectors:
                    pdf.multi_cell(180, 6, txt=f"    • {sector}: Currently {weight:.2%} of your equity allocation")
        
        # Check for concentration risk
        if top_holdings_weight > 0.5:
            pdf.ln(5)
            pdf.cell(10, 10, txt="➤", ln=False)
            pdf.multi_cell(180, 10, txt="Your portfolio has high concentration in top holdings. Consider reducing " +
                                      "individual position sizes to reduce company-specific risk.")
        
        # Risk-based recommendations
        pdf.ln(10)
        pdf.set_font("Arial", 'B', size=14)
        pdf.cell(190, 10, txt="Risk Management Recommendations", ln=True)
        
        pdf.set_font("Arial", size=11)
        
        # Risk-specific recommendations based on risk profile
        if risk_profile == "Conservative":
            if expected_return < 0.04:
                pdf.ln(2)
                pdf.cell(10, 10, txt="➤", ln=False)
                pdf.multi_cell(180, 10, txt="Consider adding some growth assets to improve returns while maintaining " +
                                          "your conservative risk profile.")
            
            if max_drawdown > 0.15:
                pdf.ln(5)
                pdf.cell(10, 10, txt="➤", ln=False)
                pdf.multi_cell(180, 10, txt="Your maximum drawdown is higher than typical for a conservative portfolio. " +
                                          "Consider adding more defensive assets to reduce potential losses in market downturns.")
        
        elif risk_profile == "Moderate":
            if sharpe_ratio < 0.4:
                pdf.ln(2)
                pdf.cell(10, 10, txt="➤", ln=False)
                pdf.multi_cell(180, 10, txt="Your portfolio's risk-adjusted return (Sharpe ratio) could be improved. " +
                                          "Consider rebalancing to optimize the risk-return tradeoff.")
        
        elif risk_profile == "Aggressive":
            if max_drawdown > 0.35:
                pdf.ln(2)
                pdf.cell(10, 10, txt="➤", ln=False)
                pdf.multi_cell(180, 10, txt="Your portfolio's maximum drawdown is very high. Consider adding some " +
                                          "defensive positions to moderate extreme losses during market downturns.")
        
        # General portfolio recommendations
        pdf.ln(10)
        pdf.set_font("Arial", 'B', size=14)
        pdf.cell(190, 10, txt="General Portfolio Recommendations", ln=True)
        
        pdf.set_font("Arial", size=11)
        pdf.ln(2)
        pdf.cell(10, 10, txt="➤", ln=False)
        pdf.multi_cell(180, 10, txt="Schedule regular portfolio reviews (at least quarterly) to ensure your investments " +
                                  "remain aligned with your financial goals.")
        
        pdf.ln(5)
        pdf.cell(10, 10, txt="➤", ln=False)
        pdf.multi_cell(180, 10, txt="Consider setting up automatic rebalancing to maintain your target asset allocation " +
                                  "as market movements change the relative weights of your holdings.")
        
        pdf.ln(5)
        pdf.cell(10, 10, txt="➤", ln=False)
        pdf.multi_cell(180, 10, txt="For more personalized recommendations including tax optimization strategies " +
                                  "and advanced scenario analysis, consider upgrading to our Premium Report.")
        
        #-----------------------
        # 6. Disclaimer
        #-----------------------
        pdf.add_page()
        pdf.set_fill_color(*self.colors['header_bg'])
        pdf.set_font("Arial", 'B', size=16)
        pdf.cell(190, 10, txt="6. Disclaimer & Legal Information", ln=True, fill=True)
        pdf.ln(5)
        
        pdf.set_font("Arial", 'B', size=12)
        pdf.cell(190, 10, txt="Important Information", ln=True)
        
        pdf.set_font("Arial", size=10)
        disclaimer = """
        This report is for informational purposes only and does not constitute investment advice, a recommendation or solicitation to buy or sell any security. The information contained in this report has been obtained from sources believed to be reliable, but ISO Financial does not warrant its completeness or accuracy.

        INVESTMENT RISKS:
        All investments involve risk, including the possible loss of principal. Past performance does not guarantee future results. The investment return and principal value of an investment will fluctuate so that an investor's shares, when redeemed, may be worth more or less than their original cost.

        SIMULATION LIMITATIONS:
        The projections or other information generated by this report regarding the likelihood of various investment outcomes are hypothetical in nature, do not reflect actual investment results, and are not guarantees of future results. The simulations are based on assumptions, and there can be no assurance that the projected or simulated results will be achieved or sustained.

        FORWARD-LOOKING STATEMENTS:
        This report may contain forward-looking statements. Forward-looking statements are not guarantees of future performance and involve risks and uncertainties. Actual results may differ materially from those projected in the forward-looking statements.

        TAX CONSIDERATIONS:
        ISO Financial does not provide tax, legal, or accounting advice. This material has been prepared for informational purposes only and is not intended to provide, and should not be relied on for, tax, legal, or accounting advice. You should consult your own tax, legal, and accounting advisors before engaging in any transaction.

        THIRD-PARTY INFORMATION:
        This report may contain data or information obtained from third-party sources. While we believe the third-party information is reliable, we have not independently verified the accuracy or completeness of the third-party information.

        By using this report, you acknowledge that you have read and understood this disclaimer and agree to its terms.
        """
        
        pdf.set_font("Arial", size=9)
        for line in disclaimer.strip().split('\n'):
            if line.strip():
                pdf.multi_cell(190, 5, txt=line.strip())
            else:
                pdf.ln(3)
        
        # Add footer with copyright and report ID
        pdf.ln(10)
        pdf.set_font("Arial", 'I', size=8)
        pdf.cell(190, 5, txt=f"Report ID: {timestamp}", ln=True)
        pdf.cell(190, 5, txt="©2025 ISO Financial. All rights reserved.", ln=True)
        
        # Output PDF
        pdf.output(output_file)
        
        return output_file
    
    def __del__(self):
        """Clean up temporary files when object is destroyed"""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {temp_file}: {str(e)}")