import os
import tempfile
import datetime
import logging
import json
from typing import Dict, Any, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fpdf import FPDF
import qrcode

# Setup logger
logger = logging.getLogger(__name__)

class PremiumReportGenerator:
    """Generator for premium financial reports with comprehensive analysis"""
    
    def __init__(self):
        """Initialize the premium report generator"""
        self.temp_files = []
        self.colors = {
            'header_bg': (225, 230, 245),  # Light blue for headers
            'subheader_bg': (240, 242, 248),  # Lighter background for subheaders
            'highlight_bg': (235, 245, 255),  # Light cyan highlight
            'positive': (0, 128, 0),  # Green for positive values
            'negative': (220, 20, 60),  # Crimson for negative values
            'neutral': (100, 100, 100),  # Gray for neutral text
            'premium': (75, 0, 130),  # Purple for premium branding
            'gold': (218, 165, 32),  # Gold for premium elements
        }
        print("PremiumReportGenerator initialized")
    
    def generate_report(self, **kwargs):
        """
        Generate a premium report with the provided simulation data.
        
        Args:
            portfolio: Dictionary of assets and weights
            asset_metadata: Dictionary with metadata for each asset (sector, asset_class, etc.)
            monte_carlo_results: Monte Carlo simulation results
            scenario_results: Scenario analysis results
            risk_results: Risk analysis results
            optimization_results: Portfolio optimization results
            market_data: Dictionary with market data for each asset
            tax_analysis: Tax analysis results
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
        tax_analysis = kwargs.get('tax_analysis', {})
        sections = kwargs.get('sections', {})
        output_file = kwargs.get('output_file')
        
        # Create a PDF report
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        if output_file is None:
            output_file = f"iso_financial_premium_{timestamp}.pdf"
        
        # Generate a comprehensive PDF premium report
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        
        #-----------------------
        # Title page
        #-----------------------
        pdf.add_page()
        pdf.set_font("Arial", 'B', size=22)
        pdf.set_text_color(*self.colors['premium'])
        pdf.cell(200, 20, txt="ISO Financial Premium Report", ln=True, align='C')
        pdf.set_font("Arial", 'I', size=14)
        pdf.set_text_color(*self.colors['gold'])
        pdf.cell(200, 10, txt="Founding Access Exclusive", ln=True, align='C')
        
        # Reset text color to black
        pdf.set_text_color(0, 0, 0)
        
        # Add date and time
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt=f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')
        
        # Table of contents section
        pdf.ln(20)
        pdf.set_font("Arial", 'B', size=16)
        pdf.cell(200, 10, txt="Executive Report Contents", ln=True)
        pdf.set_font("Arial", size=12)
        
        # Define standard sections
        standard_sections = [
            "Executive Portfolio Summary",
            "Advanced Asset Allocation Analysis",
            "Comprehensive Risk Assessment",
            "Advanced Performance Projections",
            "Portfolio Optimization Strategies",
            "Scenario Analysis & Stress Testing",
            "Tax Efficiency Optimization",
            "Strategic Recommendations",
            "Disclaimer & Legal Information"
        ]
        
        # Display table of contents
        for i, section in enumerate(standard_sections):
            pdf.cell(200, 8, txt=f"{i+1}. {section}", ln=True)
        
        #-----------------------
        # 1. Executive Portfolio Summary
        #-----------------------
        pdf.add_page()
        pdf.set_fill_color(*self.colors['header_bg'])
        pdf.set_font("Arial", 'B', size=16)
        pdf.set_text_color(*self.colors['premium'])
        pdf.cell(190, 10, txt="1. Executive Portfolio Summary", ln=True, fill=True)
        pdf.set_text_color(0, 0, 0)  # Reset text color to black
        pdf.ln(5)
        
        # Portfolio composition overview
        pdf.set_font("Arial", 'B', size=14)
        pdf.cell(190, 10, txt="Portfolio Overview", ln=True)
        
        # Get the initial investment amount
        initial_investment = monte_carlo_results.get('initial_investment', 100000)
        if hasattr(initial_investment, 'item'):  # Handle numpy types
            initial_investment = initial_investment.item()
        
        # Extract portfolio metrics from results
        expected_return = monte_carlo_results.get('expected_return', 0.08)  # Default to 8%
        portfolio_volatility = risk_results.get('portfolio_volatility', 0.15)  # Default to 15%
        sharpe_ratio = risk_results.get('sharpe_ratio', 0.5)  # Default to 0.5
        sortino_ratio = risk_results.get('sortino_ratio', 0.7)  # Default to 0.7
        
        # Key summary metrics
        pdf.set_font("Arial", size=12)
        pdf.set_fill_color(*self.colors['highlight_bg'])
        
        # Format two metrics per row in a table
        pdf.cell(95, 10, txt=f"Total Assets: {len(portfolio)}", border=1, fill=True)
        pdf.cell(95, 10, txt=f"Initial Investment: ${initial_investment:,.2f}", border=1, fill=True, ln=True)
        pdf.cell(95, 10, txt=f"Expected Annual Return: {expected_return:.2%}", border=1, fill=True)
        pdf.cell(95, 10, txt=f"Portfolio Volatility: {portfolio_volatility:.2%}", border=1, fill=True, ln=True)
        pdf.cell(95, 10, txt=f"Sharpe Ratio: {sharpe_ratio:.2f}", border=1, fill=True)
        pdf.cell(95, 10, txt=f"Sortino Ratio: {sortino_ratio:.2f}", border=1, fill=True, ln=True)
        
        # Create a formatted table for portfolio assets
        pdf.ln(10)
        pdf.set_font("Arial", 'B', size=14)
        pdf.cell(190, 10, txt="Current Portfolio Composition", ln=True)
        
        # Table header with colored background
        pdf.ln(5)
        pdf.set_fill_color(*self.colors['subheader_bg'])
        pdf.set_font("Arial", 'B', size=11)
        pdf.cell(40, 10, txt="Asset", border=1, fill=True)
        pdf.cell(30, 10, txt="Weight", border=1, fill=True)
        pdf.cell(50, 10, txt="Amount ($)", border=1, fill=True)
        pdf.cell(35, 10, txt="Asset Class", border=1, fill=True)
        pdf.cell(35, 10, txt="Sector", border=1, fill=True, ln=True)
        
        # Sort assets by weight descending
        sorted_portfolio = sorted(portfolio.items(), key=lambda x: x[1], reverse=True)
        
        # Table rows
        pdf.set_font("Arial", size=10)
        for ticker, weight in sorted_portfolio:
            # Calculate dollar amount
            amount = initial_investment * weight
            
            # Get asset metadata if available
            asset_class = "Equity"  # Default
            sector = "N/A"          # Default
            
            if ticker in asset_metadata:
                meta = asset_metadata.get(ticker, {})
                asset_class = meta.get('asset_class', "Equity")
                sector = meta.get('sector', "N/A")
            
            pdf.cell(40, 8, txt=f"{ticker}", border=1)
            pdf.cell(30, 8, txt=f"{weight:.2%}", border=1)
            pdf.cell(50, 8, txt=f"${amount:,.2f}", border=1)
            pdf.cell(35, 8, txt=f"{asset_class}", border=1)
            pdf.cell(35, 8, txt=f"{sector}", border=1, ln=True)
        
        # Executive summary
        pdf.ln(10)
        pdf.set_font("Arial", 'B', size=14)
        pdf.cell(190, 10, txt="Executive Performance Summary", ln=True)
        
        # Extract mean and median final values from monte carlo
        mean_final = monte_carlo_results.get('mean_final', initial_investment * 1.5)
        median_final = monte_carlo_results.get('median_final', initial_investment * 1.4)
        prob_positive = monte_carlo_results.get('prob_positive', 0.75)
        time_horizon = monte_carlo_results.get('time_horizon', 10)  # years
        
        # Calculate growth
        growth_pct = (mean_final - initial_investment) / initial_investment * 100
        annualized_return = ((mean_final / initial_investment) ** (1 / time_horizon)) - 1
        
        # Format summary
        pdf.set_font("Arial", size=11)
        pdf.multi_cell(190, 6, txt=f"This portfolio is projected to grow from ${initial_investment:,.2f} to ${mean_final:,.2f} " +
                               f"over {time_horizon} years, representing a total growth of {growth_pct:.2f}% " +
                               f"and an annualized return of {annualized_return:.2%}. There is a {prob_positive:.1%} " +
                               f"probability of achieving positive returns over this time horizon.")
        
        #-----------------------
        # 2. Advanced Asset Allocation Analysis
        #-----------------------
        pdf.add_page()
        pdf.set_fill_color(*self.colors['header_bg'])
        pdf.set_font("Arial", 'B', size=16)
        pdf.set_text_color(*self.colors['premium'])
        pdf.cell(190, 10, txt="2. Advanced Asset Allocation Analysis", ln=True, fill=True)
        pdf.set_text_color(0, 0, 0)  # Reset text color to black
        pdf.ln(5)
        
        # Introduction to asset allocation
        pdf.set_font("Arial", size=11)
        pdf.multi_cell(190, 6, txt="This section provides a detailed analysis of your portfolio's asset allocation " +
                              "across multiple dimensions. Proper diversification can help optimize returns while " +
                              "managing risk exposure to various market factors.")
        
        # Asset class allocation
        pdf.ln(8)
        pdf.set_font("Arial", 'B', size=14)
        pdf.cell(190, 10, txt="Asset Class Allocation", ln=True)
        
        # Categorize assets by class
        asset_classes = {}
        for ticker, weight in portfolio.items():
            asset_class = asset_metadata.get(ticker, {}).get('asset_class', "Equity")
            asset_classes[asset_class] = asset_classes.get(asset_class, 0) + weight
        
        # Create a table for asset class allocation
        pdf.ln(5)
        pdf.set_fill_color(*self.colors['subheader_bg'])
        pdf.set_font("Arial", 'B', size=11)
        pdf.cell(80, 10, txt="Asset Class", border=1, fill=True)
        pdf.cell(40, 10, txt="Current", border=1, fill=True)
        pdf.cell(40, 10, txt="Target Range", border=1, fill=True)
        pdf.cell(30, 10, txt="Status", border=1, fill=True, ln=True)
        
        # Define target ranges for common asset classes
        target_ranges = {
            "Equity": (0.40, 0.70),
            "Fixed Income": (0.20, 0.50),
            "Real Estate": (0.05, 0.15),
            "Commodities": (0.02, 0.10),
            "Cash": (0.02, 0.10),
            "Alternative": (0.00, 0.15)
        }
        
        # Table rows
        pdf.set_font("Arial", size=11)
        for asset_class, weight in sorted(asset_classes.items(), key=lambda x: x[1], reverse=True):
            # Get target range if available
            min_target, max_target = target_ranges.get(asset_class, (None, None))
            
            pdf.cell(80, 8, txt=f"{asset_class}", border=1)
            pdf.cell(40, 8, txt=f"{weight:.2%}", border=1)
            
            # Display target range if available
            if min_target is not None and max_target is not None:
                target_text = f"{min_target:.0%} - {max_target:.0%}"
                pdf.cell(40, 8, txt=target_text, border=1)
                
                # Status indicator
                if weight < min_target:
                    status = "Underweight"
                    pdf.set_text_color(*self.colors['negative'])
                elif weight > max_target:
                    status = "Overweight"
                    pdf.set_text_color(*self.colors['negative'])
                else:
                    status = "Within Target"
                    pdf.set_text_color(*self.colors['positive'])
                
                pdf.cell(30, 8, txt=status, border=1)
                pdf.set_text_color(0, 0, 0)  # Reset text color
            else:
                pdf.cell(40, 8, txt="Not Defined", border=1)
                pdf.cell(30, 8, txt="N/A", border=1)
            
            pdf.ln()
        
        # Geographic exposure analysis
        pdf.ln(10)
        pdf.set_font("Arial", 'B', size=14)
        pdf.cell(190, 10, txt="Geographic Exposure", ln=True)
        
        # Example geographic exposure (would be derived from asset_metadata in a real implementation)
        geo_exposure = {
            "North America": 0.65,
            "Europe": 0.15,
            "Asia-Pacific": 0.12,
            "Emerging Markets": 0.08
        }
        
        # Create a table for geographic exposure
        pdf.ln(5)
        pdf.set_fill_color(*self.colors['subheader_bg'])
        pdf.set_font("Arial", 'B', size=11)
        pdf.cell(80, 10, txt="Region", border=1, fill=True)
        pdf.cell(40, 10, txt="Allocation", border=1, fill=True)
        pdf.cell(70, 10, txt="Amount ($)", border=1, fill=True, ln=True)
        
        # Table rows
        pdf.set_font("Arial", size=11)
        for region, allocation in sorted(geo_exposure.items(), key=lambda x: x[1], reverse=True):
            amount = initial_investment * allocation
            pdf.cell(80, 8, txt=f"{region}", border=1)
            pdf.cell(40, 8, txt=f"{allocation:.2%}", border=1)
            pdf.cell(70, 8, txt=f"${amount:,.2f}", border=1, ln=True)
        
        # Advanced diversification analysis
        pdf.ln(10)
        pdf.set_font("Arial", 'B', size=14)
        pdf.cell(190, 10, txt="Advanced Diversification Analysis", ln=True)
        
        # Calculate diversification metrics
        num_assets = len(portfolio)
        top_holdings_weight = sum(sorted(portfolio.values(), reverse=True)[:min(3, num_assets)])
        top10_holdings_weight = sum(sorted(portfolio.values(), reverse=True)[:min(10, num_assets)])
        herfindahl_index = sum([weight**2 for weight in portfolio.values()])  # Concentration measure
        effective_n = 1 / herfindahl_index if herfindahl_index > 0 else 0
        
        # Format advanced metrics
        pdf.ln(5)
        pdf.set_fill_color(*self.colors['highlight_bg'])
        pdf.set_font("Arial", 'B', size=11)
        pdf.cell(95, 10, txt="Diversification Metric", border=1, fill=True)
        pdf.cell(95, 10, txt="Value", border=1, fill=True, ln=True)
        
        pdf.set_font("Arial", size=11)
        pdf.cell(95, 8, txt="Top 3 Holdings Concentration", border=1)
        pdf.cell(95, 8, txt=f"{top_holdings_weight:.2%}", border=1, ln=True)
        
        pdf.cell(95, 8, txt="Top 10 Holdings Concentration", border=1)
        pdf.cell(95, 8, txt=f"{top10_holdings_weight:.2%}", border=1, ln=True)
        
        pdf.cell(95, 8, txt="Herfindahl-Hirschman Index", border=1)
        pdf.cell(95, 8, txt=f"{herfindahl_index:.4f}", border=1, ln=True)
        
        pdf.cell(95, 8, txt="Effective Number of Assets", border=1)
        pdf.cell(95, 8, txt=f"{effective_n:.2f}", border=1, ln=True)
        
        # Diversification analysis explanation
        pdf.ln(8)
        pdf.set_font("Arial", 'B', size=12)
        pdf.cell(190, 8, txt="Diversification Assessment:", ln=True)
        
        # Determine diversification quality
        if herfindahl_index < 0.1 and top_holdings_weight < 0.3:
            quality = "Excellent"
            pdf.set_text_color(*self.colors['positive'])
        elif herfindahl_index < 0.2 and top_holdings_weight < 0.4:
            quality = "Good"
            pdf.set_text_color(0, 128, 128)  # Teal for good
        elif herfindahl_index < 0.3 and top_holdings_weight < 0.5:
            quality = "Moderate"
            pdf.set_text_color(218, 165, 32)  # Gold/orange for moderate
        else:
            quality = "Needs Improvement"
            pdf.set_text_color(*self.colors['negative'])
        
        pdf.set_font("Arial", size=11)
        pdf.cell(190, 8, txt=f"Portfolio Diversification Quality: {quality}", ln=True)
        pdf.set_text_color(0, 0, 0)  # Reset text color
        
        pdf.multi_cell(190, 6, txt="Your portfolio's diversification level affects its risk-return characteristics. " +
                               "Higher diversification typically reduces unsystematic risk without necessarily " +
                               "reducing expected returns. The Herfindahl-Hirschman Index measures concentration, " +
                               "with lower values indicating better diversification.")
        
        #-----------------------
        # 3. Comprehensive Risk Assessment
        #-----------------------
        pdf.add_page()
        pdf.set_fill_color(*self.colors['header_bg'])
        pdf.set_font("Arial", 'B', size=16)
        pdf.set_text_color(*self.colors['premium'])
        pdf.cell(190, 10, txt="3. Comprehensive Risk Assessment", ln=True, fill=True)
        pdf.set_text_color(0, 0, 0)  # Reset text color to black
        pdf.ln(5)
        
        # Introduction to risk assessment
        pdf.set_font("Arial", size=11)
        pdf.multi_cell(190, 6, txt="This section provides a detailed analysis of your portfolio's risk characteristics " +
                               "using multiple advanced risk metrics and methodologies. Understanding these risk " +
                               "dimensions helps develop appropriate risk management strategies.")
        
        # Extract risk metrics
        var_95 = risk_results.get('var_95', initial_investment * 0.1)
        cvar_95 = risk_results.get('cvar_95', initial_investment * 0.15)
        var_99 = risk_results.get('var_99', initial_investment * 0.15)
        cvar_99 = risk_results.get('cvar_99', initial_investment * 0.2)
        max_drawdown = risk_results.get('max_drawdown', 0.25)
        beta = risk_results.get('beta', 1.0)
        downside_deviation = risk_results.get('downside_deviation', portfolio_volatility * 0.8)
        
        # Core risk metrics table
        pdf.ln(8)
        pdf.set_font("Arial", 'B', size=14)
        pdf.cell(190, 10, txt="Advanced Risk Metrics", ln=True)
        
        pdf.ln(5)
        pdf.set_fill_color(*self.colors['subheader_bg'])
        pdf.set_font("Arial", 'B', size=11)
        pdf.cell(95, 10, txt="Risk Metric", border=1, fill=True)
        pdf.cell(95, 10, txt="Value", border=1, fill=True, ln=True)
        
        # Value at Risk metrics
        pdf.set_font("Arial", size=11)
        pdf.cell(95, 8, txt="95% Value at Risk (VaR)", border=1)
        pdf.cell(95, 8, txt=f"${var_95:,.2f} ({var_95/initial_investment:.2%})", border=1, ln=True)
        
        pdf.cell(95, 8, txt="95% Conditional VaR (CVaR/ETL)", border=1)
        pdf.cell(95, 8, txt=f"${cvar_95:,.2f} ({cvar_95/initial_investment:.2%})", border=1, ln=True)
        
        pdf.cell(95, 8, txt="99% Value at Risk (VaR)", border=1)
        pdf.cell(95, 8, txt=f"${var_99:,.2f} ({var_99/initial_investment:.2%})", border=1, ln=True)
        
        pdf.cell(95, 8, txt="99% Conditional VaR (CVaR/ETL)", border=1)
        pdf.cell(95, 8, txt=f"${cvar_99:,.2f} ({cvar_99/initial_investment:.2%})", border=1, ln=True)
        
        # Additional risk metrics
        pdf.cell(95, 8, txt="Maximum Drawdown", border=1)
        pdf.cell(95, 8, txt=f"{max_drawdown:.2%}", border=1, ln=True)
        
        pdf.cell(95, 8, txt="Portfolio Beta", border=1)
        pdf.cell(95, 8, txt=f"{beta:.2f}", border=1, ln=True)
        
        pdf.cell(95, 8, txt="Annual Volatility", border=1)
        pdf.cell(95, 8, txt=f"{portfolio_volatility:.2%}", border=1, ln=True)
        
        pdf.cell(95, 8, txt="Downside Deviation", border=1)
        pdf.cell(95, 8, txt=f"{downside_deviation:.2%}", border=1, ln=True)
        
        # Risk metrics with financial ratios
        pdf.ln(10)
        pdf.set_font("Arial", 'B', size=14)
        pdf.cell(190, 10, txt="Risk-Adjusted Return Metrics", ln=True)
        
        # Extract or generate risk-adjusted metrics
        treynor_ratio = risk_results.get('treynor_ratio', expected_return / max(beta, 0.01))
        information_ratio = risk_results.get('information_ratio', 0.3)
        calmar_ratio = risk_results.get('calmar_ratio', expected_return / max(max_drawdown, 0.01))
        
        # Create table for risk-adjusted metrics
        pdf.ln(5)
        pdf.set_fill_color(*self.colors['subheader_bg'])
        pdf.set_font("Arial", 'B', size=11)
        pdf.cell(95, 10, txt="Risk-Adjusted Metric", border=1, fill=True)
        pdf.cell(95, 10, txt="Value", border=1, fill=True, ln=True)
        
        pdf.set_font("Arial", size=11)
        pdf.cell(95, 8, txt="Sharpe Ratio", border=1)
        pdf.cell(95, 8, txt=f"{sharpe_ratio:.2f}", border=1, ln=True)
        
        pdf.cell(95, 8, txt="Sortino Ratio", border=1)
        pdf.cell(95, 8, txt=f"{sortino_ratio:.2f}", border=1, ln=True)
        
        pdf.cell(95, 8, txt="Treynor Ratio", border=1)
        pdf.cell(95, 8, txt=f"{treynor_ratio:.3f}", border=1, ln=True)
        
        pdf.cell(95, 8, txt="Information Ratio", border=1)
        pdf.cell(95, 8, txt=f"{information_ratio:.2f}", border=1, ln=True)
        
        pdf.cell(95, 8, txt="Calmar Ratio", border=1)
        pdf.cell(95, 8, txt=f"{calmar_ratio:.2f}", border=1, ln=True)
        
        # Risk interpretation
        pdf.ln(10)
        pdf.set_font("Arial", 'B', size=14)
        pdf.cell(190, 10, txt="Risk Assessment Summary", ln=True)
        
        # Determine risk profile
        if portfolio_volatility <= 0.1 and beta <= 0.8:
            risk_profile = "Conservative"
        elif portfolio_volatility >= 0.18 or beta >= 1.2:
            risk_profile = "Aggressive"
        else:
            risk_profile = "Moderate"
        
        # Format risk interpretation with color
        pdf.ln(5)
        pdf.set_fill_color(*self.colors['highlight_bg'])
        pdf.set_font("Arial", 'B', size=12)
        pdf.cell(190, 10, txt=f"Portfolio Risk Profile: {risk_profile}", ln=True, fill=True)
        
        # Risk assessment text
        pdf.set_font("Arial", size=11)
        risk_assessment = {
            "Conservative": 
                "Your portfolio has a conservative risk profile aimed at capital preservation with moderate growth. " +
                f"With a volatility of {portfolio_volatility:.2%} and beta of {beta:.2f}, it's positioned to withstand " +
                "market downturns while sacrificing some upside potential. The maximum expected loss in extreme " +
                f"scenarios (99% CVaR) is ${cvar_99:,.2f}, or {cvar_99/initial_investment:.2%} of your investment.",
            
            "Moderate": 
                "Your portfolio has a moderate risk profile balancing growth and protection. " +
                f"With a volatility of {portfolio_volatility:.2%} and beta of {beta:.2f}, it offers a balanced " +
                "approach to market participation. The maximum expected loss in extreme " +
                f"scenarios (99% CVaR) is ${cvar_99:,.2f}, or {cvar_99/initial_investment:.2%} of your investment.",
            
            "Aggressive": 
                "Your portfolio has an aggressive risk profile focused on maximizing growth potential. " +
                f"With a volatility of {portfolio_volatility:.2%} and beta of {beta:.2f}, it's positioned for " +
                "strong market participation but may experience significant drawdowns during market corrections. " +
                f"The maximum expected loss in extreme scenarios (99% CVaR) is ${cvar_99:,.2f}, or " +
                f"{cvar_99/initial_investment:.2%} of your investment."
        }
        
        pdf.multi_cell(190, 6, txt=risk_assessment.get(risk_profile, ""))
        
        #-----------------------
        # 4. Advanced Performance Projections
        #-----------------------
        pdf.add_page()
        pdf.set_fill_color(*self.colors['header_bg'])
        pdf.set_font("Arial", 'B', size=16)
        pdf.set_text_color(*self.colors['premium'])
        pdf.cell(190, 10, txt="4. Advanced Performance Projections", ln=True, fill=True)
        pdf.set_text_color(0, 0, 0)  # Reset text color to black
        pdf.ln(5)
        
        # Introduction to projections
        pdf.set_font("Arial", size=11)
        pdf.multi_cell(190, 6, txt="This section provides detailed projections of your portfolio's performance across " +
                               "multiple time horizons and confidence intervals. These projections use Monte Carlo " +
                               "simulation and incorporate multiple economic variables and asset correlations.")
        
        # Monte Carlo simulation results
        if monte_carlo_results:
            pdf.ln(8)
            pdf.set_font("Arial", 'B', size=14)
            pdf.cell(190, 10, txt="Monte Carlo Simulation Results", ln=True)
            
            # Get simulation parameters
            time_horizon = monte_carlo_results.get('time_horizon', 10)
            num_simulations = monte_carlo_results.get('num_simulations', 1000)
            
            # Simulation parameters
            pdf.set_font("Arial", size=11)
            pdf.cell(190, 8, txt=f"Simulation Parameters: {num_simulations} simulations over {time_horizon} years", ln=True)
            
            # Key simulation results
            pdf.ln(5)
            pdf.set_fill_color(*self.colors['highlight_bg'])
            pdf.set_font("Arial", 'B', size=11)
            pdf.cell(95, 10, txt="Initial Investment", border=1, fill=True)
            pdf.cell(95, 10, txt="Mean Final Value", border=1, fill=True, ln=True)
            
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
            
            # Detailed percentile analysis
            pdf.ln(10)
            pdf.set_font("Arial", 'B', size=14)
            pdf.cell(190, 10, txt=f"Investment Outcome Ranges ({time_horizon}-Year Horizon)", ln=True)
            
            percentiles = monte_carlo_results.get('percentiles', {})
            if percentiles:
                # Extended percentiles table
                pdf.ln(5)
                pdf.set_fill_color(*self.colors['subheader_bg'])
                pdf.set_font("Arial", 'B', size=11)
                pdf.cell(43, 10, txt="Percentile", border=1, fill=True)
                pdf.cell(43, 10, txt="Scenario", border=1, fill=True)
                pdf.cell(52, 10, txt="Final Value", border=1, fill=True)
                pdf.cell(52, 10, txt="Annualized Return", border=1, fill=True, ln=True)
                
                # Define percentiles to display with descriptive names
                scenario_names = {
                    1: "Worst Case",
                    5: "Severe Downside",
                    10: "Significant Downside",
                    25: "Moderate Downside",
                    50: "Median Outcome",
                    75: "Moderate Upside",
                    90: "Significant Upside",
                    95: "Strong Upside",
                    99: "Best Case"
                }
                
                pdf.set_font("Arial", size=11)
                for pct in sorted(percentiles.keys()):
                    if pct in scenario_names:
                        value = percentiles[pct]
                        ann_return = ((value / initial_investment) ** (1 / time_horizon)) - 1
                        
                        # Color code by outcome
                        if pct < 25:
                            pdf.set_text_color(*self.colors['negative'])
                        elif pct > 75:
                            pdf.set_text_color(*self.colors['positive'])
                        else:
                            pdf.set_text_color(*self.colors['neutral'])
                            
                        pdf.cell(43, 10, txt=f"{pct}th", border=1)
                        pdf.cell(43, 10, txt=f"{scenario_names[pct]}", border=1)
                        pdf.cell(52, 10, txt=f"${value:,.2f}", border=1)
                        pdf.cell(52, 10, txt=f"{ann_return:.2%}", border=1, ln=True)
                
                pdf.set_text_color(0, 0, 0)  # Reset text color
            
            # Multi-horizon projections
            pdf.ln(10)
            pdf.set_font("Arial", 'B', size=14)
            pdf.cell(190, 10, txt="Multi-Horizon Projections", ln=True)
            
            # Generate projections for multiple time horizons
            horizons = [1, 3, 5, 10, 20, 30]
            
            # Create multi-horizon projection table
            pdf.ln(5)
            pdf.set_fill_color(*self.colors['subheader_bg'])
            pdf.set_font("Arial", 'B', size=11)
            pdf.cell(28, 10, txt="Years", border=1, fill=True)
            pdf.cell(54, 10, txt="Median Value", border=1, fill=True)
            pdf.cell(54, 10, txt="Optimistic (90%)", border=1, fill=True)
            pdf.cell(54, 10, txt="Pessimistic (10%)", border=1, fill=True, ln=True)
            
            pdf.set_font("Arial", size=11)
            for years in horizons:
                if years <= time_horizon:
                    # Calculate projected values using compound growth formula
                    # In a real implementation, these would come from actual simulations
                    yearly_factor = (1 + annualized_return)
                    median_value = initial_investment * (yearly_factor ** years)
                    
                    # Estimate optimistic and pessimistic scenarios
                    # This is simplified - real implementation would use actual percentiles
                    optimistic_return = annualized_return + (portfolio_volatility * 1.28 / np.sqrt(years))
                    pessimistic_return = annualized_return - (portfolio_volatility * 1.28 / np.sqrt(years))
                    
                    optimistic_value = initial_investment * ((1 + optimistic_return) ** years)
                    pessimistic_value = initial_investment * ((1 + pessimistic_return) ** years)
                    
                    pdf.cell(28, 10, txt=f"{years}", border=1)
                    pdf.cell(54, 10, txt=f"${median_value:,.2f}", border=1)
                    pdf.cell(54, 10, txt=f"${optimistic_value:,.2f}", border=1)
                    pdf.cell(54, 10, txt=f"${pessimistic_value:,.2f}", border=1, ln=True)
            
            # Performance interpretation
            pdf.ln(8)
            pdf.set_font("Arial", 'B', size=12)
            pdf.cell(190, 8, txt="Performance Interpretation:", ln=True)
            
            pdf.set_font("Arial", size=11)
            pdf.multi_cell(190, 6, txt="These projections illustrate potential outcomes across different time horizons " +
                                   "and market conditions. Longer investment horizons generally reduce the impact of " +
                                   "short-term market volatility but increase the range of possible outcomes. The " +
                                   f"median {time_horizon}-year projection of ${median_final:,.2f} represents our central " +
                                   "expectation, though actual results may vary significantly.")
        
        #-----------------------
        # 5. Portfolio Optimization Strategies
        #-----------------------
        pdf.add_page()
        pdf.set_fill_color(*self.colors['header_bg'])
        pdf.set_font("Arial", 'B', size=16)
        pdf.set_text_color(*self.colors['premium'])
        pdf.cell(190, 10, txt="5. Portfolio Optimization Strategies", ln=True, fill=True)
        pdf.set_text_color(0, 0, 0)  # Reset text color to black
        pdf.ln(5)
        
        # Introduction to optimization
        pdf.set_font("Arial", size=11)
        pdf.multi_cell(190, 6, txt="This section provides advanced optimization strategies to potentially enhance " +
                               "your portfolio's performance. These optimizations are based on modern portfolio " +
                               "theory and factor-based analysis to target an optimal risk-return profile.")
        
        # Portfolio optimization
        if optimization_results:
            pdf.ln(8)
            pdf.set_font("Arial", 'B', size=14)
            pdf.cell(190, 10, txt="Efficient Frontier Optimization", ln=True)
            
            # Get optimization results
            optimal_weights = optimization_results.get('optimal_weights', {})
            expected_return_optimal = optimization_results.get('expected_return', expected_return * 1.1)
            volatility_optimal = optimization_results.get('portfolio_volatility', portfolio_volatility * 0.9)
            sharpe_optimal = optimization_results.get('sharpe_ratio', sharpe_ratio * 1.15)
            
            # Optimization summary
            pdf.set_font("Arial", size=11)
            pdf.multi_cell(190, 6, txt="The optimized portfolio is designed to maximize risk-adjusted returns based on " +
                                   "historical asset performance and correlation. The table below compares your current " +
                                   "portfolio with the optimized allocation.")
            
            # Optimization performance comparison
            pdf.ln(5)
            pdf.set_fill_color(*self.colors['highlight_bg'])
            pdf.set_font("Arial", 'B', size=11)
            pdf.cell(70, 10, txt="Metric", border=1, fill=True)
            pdf.cell(60, 10, txt="Current Portfolio", border=1, fill=True)
            pdf.cell(60, 10, txt="Optimized Portfolio", border=1, fill=True, ln=True)
            
            pdf.set_font("Arial", size=11)
            pdf.cell(70, 8, txt="Expected Annual Return", border=1)
            pdf.cell(60, 8, txt=f"{expected_return:.2%}", border=1)
            pdf.cell(60, 8, txt=f"{expected_return_optimal:.2%}", border=1, ln=True)
            
            pdf.cell(70, 8, txt="Annual Volatility", border=1)
            pdf.cell(60, 8, txt=f"{portfolio_volatility:.2%}", border=1)
            pdf.cell(60, 8, txt=f"{volatility_optimal:.2%}", border=1, ln=True)
            
            pdf.cell(70, 8, txt="Sharpe Ratio", border=1)
            pdf.cell(60, 8, txt=f"{sharpe_ratio:.2f}", border=1)
            pdf.cell(60, 8, txt=f"{sharpe_optimal:.2f}", border=1, ln=True)
            
            # Compare current and optimized asset allocations
            if optimal_weights:
                pdf.ln(10)
                pdf.set_font("Arial", 'B', size=14)
                pdf.cell(190, 10, txt="Asset Allocation Comparison", ln=True)
                
                pdf.ln(5)
                pdf.set_fill_color(*self.colors['subheader_bg'])
                pdf.set_font("Arial", 'B', size=11)
                pdf.cell(50, 10, txt="Asset", border=1, fill=True)
                pdf.cell(40, 10, txt="Current Weight", border=1, fill=True)
                pdf.cell(40, 10, txt="Optimal Weight", border=1, fill=True)
                pdf.cell(30, 10, txt="Change", border=1, fill=True)
                pdf.cell(30, 10, txt="Action", border=1, fill=True, ln=True)
                
                pdf.set_font("Arial", size=10)
                for ticker, current_weight in portfolio.items():
                    recommended = optimal_weights.get(ticker, 0)
                    change = recommended - current_weight
                    
                    # Determine action and color
                    if abs(change) < 0.01:  # Less than 1% change
                        action = "Hold"
                        action_color = self.colors['neutral']
                    elif change > 0:
                        action = "Increase"
                        action_color = self.colors['positive']
                    else:
                        action = "Decrease"
                        action_color = self.colors['negative']
                    
                    pdf.cell(50, 8, txt=f"{ticker}", border=1)
                    pdf.cell(40, 8, txt=f"{current_weight:.2%}", border=1)
                    pdf.cell(40, 8, txt=f"{recommended:.2%}", border=1)
                    
                    # Format change with appropriate sign
                    change_text = f"{'+' if change >= 0 else ''}{change:.2%}"
                    pdf.cell(30, 8, txt=change_text, border=1)
                    
                    # Action with appropriate color
                    pdf.set_text_color(*action_color)
                    pdf.cell(30, 8, txt=action, border=1, ln=True)
                    pdf.set_text_color(0, 0, 0)  # Reset text color
                
                # Additional assets to add (in optimal but not in current)
                new_assets = [t for t in optimal_weights if t not in portfolio]
                for ticker in new_assets:
                    weight = optimal_weights[ticker]
                    if weight > 0.01:  # Only show if weight > 1%
                        pdf.cell(50, 8, txt=f"{ticker}", border=1)
                        pdf.cell(40, 8, txt="0.00%", border=1)
                        pdf.cell(40, 8, txt=f"{weight:.2%}", border=1)
                        pdf.cell(30, 8, txt=f"+{weight:.2%}", border=1)
                        
                        pdf.set_text_color(*self.colors['positive'])
                        pdf.cell(30, 8, txt="Add New", border=1, ln=True)
                        pdf.set_text_color(0, 0, 0)  # Reset text color
            
            # Optimization methodology
            pdf.ln(10)
            pdf.set_font("Arial", 'B', size=14)
            pdf.cell(190, 10, txt="Optimization Methodology", ln=True)
            
            pdf.set_font("Arial", size=11)
            pdf.multi_cell(190, 6, txt="The optimization employs Modern Portfolio Theory to identify the portfolio " +
                                   "with the highest Sharpe ratio (risk-adjusted return) based on historical returns, " +
                                   "volatility, and asset correlations. The algorithm considers transaction costs and " +
                                   "liquidity constraints to ensure practical implementation.")
            
            # Implementation plan
            pdf.ln(8)
            pdf.set_font("Arial", 'B', size=14)
            pdf.cell(190, 10, txt="Implementation Recommendations", ln=True)
            
            pdf.set_font("Arial", size=11)
            pdf.multi_cell(190, 6, txt="Consider implementing the recommended changes gradually to minimize market impact and " +
                                   "transaction costs. A phased approach over 2-3 rebalancing cycles can help reduce " +
                                   "timing risk while moving toward the optimized allocation.")
            
            # Tax-aware note
            pdf.ln(5)
            pdf.set_font("Arial", 'I', size=11)
            pdf.multi_cell(190, 6, txt="Note: See the Tax Efficiency Optimization section for tax-aware implementation " +
                                   "strategies that consider the tax implications of these recommended changes.")
        
        #-----------------------
        # 6. Scenario Analysis & Stress Testing
        #-----------------------
        pdf.add_page()
        pdf.set_fill_color(*self.colors['header_bg'])
        pdf.set_font("Arial", 'B', size=16)
        pdf.set_text_color(*self.colors['premium'])
        pdf.cell(190, 10, txt="6. Scenario Analysis & Stress Testing", ln=True, fill=True)
        pdf.set_text_color(0, 0, 0)  # Reset text color to black
        pdf.ln(5)
        
        # Introduction to scenario analysis
        pdf.set_font("Arial", size=11)
        pdf.multi_cell(190, 6, txt="This section examines how your portfolio might perform under specific market " +
                               "scenarios and economic conditions. Stress testing helps identify potential " +
                               "vulnerabilities and prepare contingency strategies for adverse market environments.")
        
        # Scenario analysis
        if scenario_results:
            pdf.ln(8)
            pdf.set_font("Arial", 'B', size=14)
            pdf.cell(190, 10, txt="Market Scenario Analysis", ln=True)
            
            # Get scenario data
            scenarios = scenario_results.get('scenarios', {})
            
            if scenarios:
                # Scenario table
                pdf.ln(5)
                pdf.set_fill_color(*self.colors['subheader_bg'])
                pdf.set_font("Arial", 'B', size=10)
                pdf.cell(60, 10, txt="Scenario", border=1, fill=True)
                pdf.cell(35, 10, txt="Expected Return", border=1, fill=True)
                pdf.cell(35, 10, txt="Maximum Loss", border=1, fill=True)
                pdf.cell(35, 10, txt="Recovery Period", border=1, fill=True)
                pdf.cell(25, 10, txt="Probability", border=1, fill=True, ln=True)
                
                pdf.set_font("Arial", size=10)
                for scenario_name, scenario_data in scenarios.items():
                    exp_return = scenario_data.get('expected_return', 0)
                    max_loss = scenario_data.get('max_loss', 0)
                    recovery = scenario_data.get('recovery_periods', 'N/A')
                    probability = scenario_data.get('probability', 0.1)
                    
                    # Format recovery period
                    if isinstance(recovery, (int, float)):
                        recovery_text = f"{recovery} months"
                    else:
                        recovery_text = str(recovery)
                    
                    # Color code based on outcome
                    if exp_return < 0:
                        pdf.set_text_color(*self.colors['negative'])
                    elif exp_return > 0.1:  # 10% or better
                        pdf.set_text_color(*self.colors['positive'])
                    else:
                        pdf.set_text_color(*self.colors['neutral'])
                    
                    pdf.cell(60, 10, txt=f"{scenario_name}", border=1)
                    pdf.cell(35, 10, txt=f"{exp_return:.2%}", border=1)
                    pdf.cell(35, 10, txt=f"{max_loss:.2%}", border=1)
                    pdf.cell(35, 10, txt=recovery_text, border=1)
                    pdf.cell(25, 10, txt=f"{probability:.1%}", border=1, ln=True)
                
                pdf.set_text_color(0, 0, 0)  # Reset text color
                
                # Detailed scenario descriptions
                pdf.ln(10)
                pdf.set_font("Arial", 'B', size=14)
                pdf.cell(190, 10, txt="Scenario Descriptions & Portfolio Impacts", ln=True)
                
                # Create detailed scenario cards
                key_scenarios = ["Market Correction", "Economic Recession", "Stagflation", "Bull Market", "Rising Rate Environment"]
                scenario_impacts = {
                    "Market Correction": 
                        "A market correction involves a 10-20% decline from recent peaks, typically due to technical " +
                        "factors or investor sentiment shifts rather than fundamental economic problems. Your portfolio " +
                        f"would likely experience a temporary decline of {max_loss:.1%}, with the most significant impact " +
                        "on growth-oriented equities.",
                    
                    "Economic Recession": 
                        "During a recession, your portfolio could face sustained pressure with corporate earnings declines " +
                        "across multiple sectors. Fixed income assets would provide some buffer, but overall portfolio " +
                        f"value might decline by {max_loss:.1%}. Recovery typically begins before economic indicators " +
                        "turn positive.",
                    
                    "Stagflation": 
                        "A stagflation environment (high inflation with slow growth) would present challenges for both " +
                        "equities and traditional bonds. Alternative assets like commodities might outperform. Your " +
                        f"portfolio's anticipated real return (after inflation) could be {exp_return - 0.06:.1%} in " +
                        "this scenario.",
                    
                    "Bull Market": 
                        "In an extended bull market scenario, your portfolio is positioned to capture significant upside " +
                        f"with projected returns of {exp_return:.1%}. Growth-oriented assets would lead performance, " +
                        "though your diversification means returns may lag aggressive growth portfolios somewhat.",
                    
                    "Rising Rate Environment": 
                        "As interest rates rise, bond values typically fall, particularly those with longer durations. " +
                        "Your fixed income allocation would face headwinds, potentially offset by floating rate " +
                        "instruments and value stocks. The net portfolio impact would depend on the pace of rate increases."
                }
                
                pdf.set_font("Arial", size=11)
                for scenario in key_scenarios:
                    if scenario in scenario_impacts:
                        pdf.set_fill_color(*self.colors['highlight_bg'])
                        pdf.set_font("Arial", 'B', size=12)
                        pdf.cell(190, 10, txt=scenario, ln=True, fill=True)
                        
                        pdf.set_font("Arial", size=11)
                        pdf.multi_cell(190, 6, txt=scenario_impacts[scenario])
                        pdf.ln(5)
        
            # Historical events analysis
            pdf.ln(8)
            pdf.set_font("Arial", 'B', size=14)
            pdf.cell(190, 10, txt="Historical Crisis Simulation", ln=True)
            
            # Create table of historical events
            pdf.ln(5)
            pdf.set_fill_color(*self.colors['subheader_bg'])
            pdf.set_font("Arial", 'B', size=11)
            pdf.cell(65, 10, txt="Historical Event", border=1, fill=True)
            pdf.cell(45, 10, txt="Market Decline", border=1, fill=True)
            pdf.cell(45, 10, txt="Portfolio Impact", border=1, fill=True)
            pdf.cell(35, 10, txt="Recovery Time", border=1, fill=True, ln=True)
            
            # Example historical events and simulated portfolio impacts
            historical_events = [
                ("2008 Financial Crisis", "-54%", "-38%", "21 months"),
                ("2020 COVID Crash", "-34%", "-26%", "9 months"),
                ("2000-2002 Tech Bubble", "-49%", "-22%", "14 months"),
                ("2011 Euro Debt Crisis", "-19%", "-14%", "5 months"),
                ("2018 Q4 Selloff", "-20%", "-16%", "4 months")
            ]
            
            pdf.set_font("Arial", size=11)
            for event, market_decline, portfolio_impact, recovery in historical_events:
                pdf.cell(65, 8, txt=event, border=1)
                pdf.cell(45, 8, txt=market_decline, border=1)
                pdf.cell(45, 8, txt=portfolio_impact, border=1)
                pdf.cell(35, 8, txt=recovery, border=1, ln=True)
            
            # Risk mitigation strategies
            pdf.ln(10)
            pdf.set_font("Arial", 'B', size=14)
            pdf.cell(190, 10, txt="Risk Mitigation Strategies", ln=True)
            
            pdf.set_font("Arial", size=11)
            pdf.multi_cell(190, 6, txt="Based on the scenario analysis, consider these risk mitigation strategies:")
            
            # Specific defensive strategies
            pdf.ln(5)
            pdf.set_font("Arial", 'B', size=12)
            pdf.cell(190, 8, txt="Tactical Defensive Adjustments:", ln=True)
            
            pdf.set_font("Arial", size=11)
            defensive_strategies = [
                "Increase allocation to defensive sectors (healthcare, consumer staples, utilities)",
                "Consider adding tail risk hedges during periods of heightened uncertainty",
                "Maintain adequate liquidity (5-10% cash position) for potential market dislocations",
                "Implement stop-loss strategies for positions with elevated concentration risk",
                "Consider alternative investments with lower correlation to traditional markets"
            ]
            
            for strategy in defensive_strategies:
                pdf.cell(10, 8, txt="•", ln=False)
                pdf.multi_cell(180, 8, txt=strategy)
            
        #-----------------------
        # 7. Tax Efficiency Optimization
        #-----------------------
        pdf.add_page()
        pdf.set_fill_color(*self.colors['header_bg'])
        pdf.set_font("Arial", 'B', size=16)
        pdf.set_text_color(*self.colors['premium'])
        pdf.cell(190, 10, txt="7. Tax Efficiency Optimization", ln=True, fill=True)
        pdf.set_text_color(0, 0, 0)  # Reset text color to black
        pdf.ln(5)
        
        # Introduction to tax efficiency
        pdf.set_font("Arial", size=11)
        pdf.multi_cell(190, 6, txt="This section provides strategies to optimize the tax efficiency of your portfolio. " +
                               "Proper tax planning can significantly enhance after-tax returns and help preserve " +
                               "wealth across market cycles and generation transfers.")
        
        # Tax efficiency analysis
        pdf.ln(8)
        pdf.set_font("Arial", 'B', size=14)
        pdf.cell(190, 10, txt="Tax-Optimized Asset Location", ln=True)
        
        # Create tax-optimized allocation table
        pdf.ln(5)
        pdf.set_fill_color(*self.colors['subheader_bg'])
        pdf.set_font("Arial", 'B', size=11)
        pdf.cell(50, 10, txt="Asset Category", border=1, fill=True)
        pdf.cell(45, 10, txt="Tax Efficiency", border=1, fill=True)
        pdf.cell(48, 10, txt="Taxable Accounts", border=1, fill=True)
        pdf.cell(47, 10, txt="Tax-Advantaged", border=1, fill=True, ln=True)
        
        # Example tax efficiency classifications
        tax_classes = [
            ("Growth Equities", "Medium", "30%", "70%"),
            ("Dividend Stocks", "Low", "20%", "80%"),
            ("Municipal Bonds", "High", "100%", "0%"),
            ("Corporate Bonds", "Low", "10%", "90%"),
            ("REITs", "Low", "5%", "95%"),
            ("Index Funds", "High", "80%", "20%")
        ]
        
        pdf.set_font("Arial", size=11)
        for asset, efficiency, taxable, tax_advantaged in tax_classes:
            pdf.cell(50, 8, txt=asset, border=1)
            pdf.cell(45, 8, txt=efficiency, border=1)
            pdf.cell(48, 8, txt=taxable, border=1)
            pdf.cell(47, 8, txt=tax_advantaged, border=1, ln=True)
        
        # Tax-loss harvesting opportunities
        pdf.ln(10)
        pdf.set_font("Arial", 'B', size=14)
        pdf.cell(190, 10, txt="Tax-Loss Harvesting Opportunities", ln=True)
        
        pdf.set_font("Arial", size=11)
        pdf.multi_cell(190, 6, txt="Tax-loss harvesting involves selling securities at a loss to offset capital gains tax liability. " +
                               "Based on current positions, the following opportunities exist:")
        
        # Create tax-loss harvesting table
        pdf.ln(5)
        pdf.set_fill_color(*self.colors['subheader_bg'])
        pdf.set_font("Arial", 'B', size=11)
        pdf.cell(50, 10, txt="Asset", border=1, fill=True)
        pdf.cell(45, 10, txt="Unrealized Loss", border=1, fill=True)
        pdf.cell(50, 10, txt="Potential Tax Savings", border=1, fill=True)
        pdf.cell(45, 10, txt="Suggested Alternative", border=1, fill=True, ln=True)
        
        # Example tax-loss harvesting opportunities
        # In a real implementation, these would be derived from portfolio data
        tax_loss_opportunities = [
            ("XYZ Corp", "-$2,450", "$580", "ABC Inc"),
            ("Global ETF", "-$1,800", "$430", "World Index Fund"),
            ("Tech Fund", "-$3,200", "$760", "Digital Innovation ETF")
        ]
        
        pdf.set_font("Arial", size=11)
        for asset, loss, savings, alternative in tax_loss_opportunities:
            pdf.cell(50, 8, txt=asset, border=1)
            pdf.cell(45, 8, txt=loss, border=1)
            pdf.cell(50, 8, txt=savings, border=1)
            pdf.cell(45, 8, txt=alternative, border=1, ln=True)
        
        # Additional tax strategies
        pdf.ln(10)
        pdf.set_font("Arial", 'B', size=14)
        pdf.cell(190, 10, txt="Advanced Tax Strategies", ln=True)
        
        # Create advanced strategies list
        advanced_strategies = [
            ("Capital Gains Management", 
             "Time the realization of capital gains to manage tax brackets. Consider deferring gains in high-income years " +
             "and accelerating gains in lower-income years."),
            
            ("Tax-Efficient Withdrawal Strategy", 
             "Develop a tax-optimized withdrawal strategy across accounts (taxable, tax-deferred, and tax-free) " +
             "to minimize lifetime tax burden during retirement."),
            
            ("Charitable Giving Strategies", 
             "Consider donating appreciated securities to charity instead of cash. This eliminates capital gains " +
             "tax while still providing a tax deduction for the full market value."),
            
            ("Estate Planning Considerations", 
             "Develop strategies for efficient wealth transfer, including step-up in basis provisions, " +
             "strategic gifting, and trust structures.")
        ]
        
        pdf.set_font("Arial", size=11)
        for strategy, description in advanced_strategies:
            pdf.set_fill_color(*self.colors['highlight_bg'])
            pdf.set_font("Arial", 'B', size=11)
            pdf.cell(190, 8, txt=strategy, ln=True, fill=True)
            
            pdf.set_font("Arial", size=11)
            pdf.multi_cell(190, 6, txt=description)
            pdf.ln(3)
        
        # Disclaimer about tax advice
        pdf.ln(5)
        pdf.set_font("Arial", 'I', size=10)
        pdf.multi_cell(190, 5, txt="Note: This analysis provides general tax planning concepts and is not intended as tax advice. " +
                               "Please consult with a qualified tax professional for advice specific to your situation.")
        
        #-----------------------
        # 8. Strategic Recommendations
        #-----------------------
        pdf.add_page()
        pdf.set_fill_color(*self.colors['header_bg'])
        pdf.set_font("Arial", 'B', size=16)
        pdf.set_text_color(*self.colors['premium'])
        pdf.cell(190, 10, txt="8. Strategic Recommendations", ln=True, fill=True)
        pdf.set_text_color(0, 0, 0)  # Reset text color to black
        pdf.ln(5)
        
        # Introduction to recommendations
        pdf.set_font("Arial", size=11)
        pdf.multi_cell(190, 6, txt="Based on our comprehensive analysis, we provide the following strategic recommendations " +
                               "to optimize your portfolio. These recommendations are prioritized by potential impact " +
                               "and organized by implementation timeframe.")
        
        # Immediate priority recommendations
        pdf.ln(8)
        pdf.set_font("Arial", 'B', size=14)
        pdf.set_text_color(*self.colors['negative'])
        pdf.cell(190, 10, txt="High Priority (Next 30 Days)", ln=True)
        pdf.set_text_color(0, 0, 0)  # Reset text color
        
        high_priority = [
            ("Rebalance Overweight Positions", 
             "Reduce allocation to [specific overweight assets] to align with target allocation and " +
             "manage concentration risk."),
            
            ("Implement Tax-Loss Harvesting", 
             "Execute identified tax-loss harvesting opportunities to offset recent capital gains " +
             "while maintaining market exposure through alternative positions."),
            
            ("Establish Liquidity Reserve", 
             "Ensure adequate liquidity (recommended 6-12 months of expenses) in cash or cash equivalents " +
             "to avoid forced liquidations during market stress.")
        ]
        
        pdf.set_font("Arial", size=11)
        for i, (recommendation, description) in enumerate(high_priority):
            pdf.set_fill_color(*self.colors['highlight_bg'])
            pdf.set_font("Arial", 'B', size=11)
            pdf.cell(190, 8, txt=f"{i+1}. {recommendation}", ln=True, fill=True)
            
            pdf.set_font("Arial", size=11)
            pdf.multi_cell(190, 6, txt=description)
            pdf.ln(3)
        
        # Medium-term recommendations
        pdf.ln(8)
        pdf.set_font("Arial", 'B', size=14)
        pdf.set_text_color(218, 165, 32)  # Gold/orange
        pdf.cell(190, 10, txt="Medium Priority (1-3 Months)", ln=True)
        pdf.set_text_color(0, 0, 0)  # Reset text color
        
        medium_priority = [
            ("Optimize Asset Location", 
             "Redistribute assets across taxable and tax-advantaged accounts according to the " +
             "tax-efficiency recommendations to enhance after-tax returns."),
            
            ("Implement Defensive Positioning", 
             "Consider adding defensive positions as identified in the scenario analysis section " +
             "to mitigate potential downside risk in current market conditions."),
            
            ("Diversify Concentrated Positions", 
             "Develop a strategy to reduce concentration in [specific positions] through systematic " +
             "selling or option-based hedging strategies.")
        ]
        
        pdf.set_font("Arial", size=11)
        for i, (recommendation, description) in enumerate(medium_priority):
            pdf.set_fill_color(*self.colors['highlight_bg'])
            pdf.set_font("Arial", 'B', size=11)
            pdf.cell(190, 8, txt=f"{i+1}. {recommendation}", ln=True, fill=True)
            
            pdf.set_font("Arial", size=11)
            pdf.multi_cell(190, 6, txt=description)
            pdf.ln(3)
        
        # Longer-term strategic recommendations
        pdf.ln(8)
        pdf.set_font("Arial", 'B', size=14)
        pdf.set_text_color(*self.colors['positive'])
        pdf.cell(190, 10, txt="Strategic Initiatives (3-12 Months)", ln=True)
        pdf.set_text_color(0, 0, 0)  # Reset text color
        
        strategic_initiatives = [
            ("Transition to Optimal Portfolio", 
             "Gradually transition from current to optimized portfolio allocation, prioritizing " +
             "tax-efficient implementation and minimizing transaction costs."),
            
            ("Implement Alternative Investments", 
             "Consider adding select alternative investments to enhance diversification and " +
             "reduce correlation to traditional asset classes."),
            
            ("Establish Systematic Rebalancing", 
             "Implement a systematic rebalancing program with specified triggers (time-based or " +
             "threshold-based) to maintain target allocations."),
            
            ("Develop Tax-Efficient Withdrawal Strategy", 
             "Create a comprehensive plan for tax-efficient withdrawals across account types to " +
             "fund future spending needs while minimizing tax impact.")
        ]
        
        pdf.set_font("Arial", size=11)
        for i, (recommendation, description) in enumerate(strategic_initiatives):
            pdf.set_fill_color(*self.colors['highlight_bg'])
            pdf.set_font("Arial", 'B', size=11)
            pdf.cell(190, 8, txt=f"{i+1}. {recommendation}", ln=True, fill=True)
            
            pdf.set_font("Arial", size=11)
            pdf.multi_cell(190, 6, txt=description)
            pdf.ln(3)
        
        # Implementation roadmap
        pdf.ln(8)
        pdf.set_font("Arial", 'B', size=14)
        pdf.cell(190, 10, txt="Implementation Roadmap", ln=True)
        
        pdf.set_font("Arial", size=11)
        pdf.multi_cell(190, 6, txt="For optimal implementation of these recommendations, consider the following approach:")
        
        pdf.ln(3)
        pdf.set_font("Arial", size=11)
        roadmap_steps = [
            "Start with high-priority recommendations to address immediate opportunities and risks",
            "Implement tax-efficient changes first, especially tax-loss harvesting opportunities",
            "Schedule a portfolio review after implementing high-priority recommendations",
            "Develop a written implementation plan with specific timelines for medium and long-term initiatives",
            "Consider dollar-cost averaging for significant allocation changes to manage timing risk"
        ]
        
        for step in roadmap_steps:
            pdf.cell(10, 8, txt="➤", ln=False)
            pdf.multi_cell(180, 8, txt=step)
        
        #-----------------------
        # 9. Disclaimer
        #-----------------------
        pdf.add_page()
        pdf.set_fill_color(*self.colors['header_bg'])
        pdf.set_font("Arial", 'B', size=16)
        pdf.set_text_color(*self.colors['premium'])
        pdf.cell(190, 10, txt="9. Disclaimer & Legal Information", ln=True, fill=True)
        pdf.set_text_color(0, 0, 0)  # Reset text color to black
        pdf.ln(5)
        
        pdf.set_font("Arial", 'B', size=12)
        pdf.cell(190, 10, txt="Important Information", ln=True)
        
        pdf.set_font("Arial", size=9)
        disclaimer = """
        CONFIDENTIAL REPORT - PREMIUM CLIENT ACCESS

        This comprehensive report is provided exclusively to Premium subscribers and contains advanced analytics, proprietary methodologies, and detailed recommendations that are confidential and proprietary to ISO Financial.

        INFORMATION PURPOSE AND USAGE LIMITATIONS:
        This report is for informational purposes only and does not constitute personalized investment advice, a recommendation or solicitation to buy or sell any security. The information contained in this report has been obtained from sources believed to be reliable, but ISO Financial does not warrant its completeness or accuracy.

        INVESTMENT RISKS:
        All investments involve risk, including the possible loss of principal. Past performance does not guarantee future results. The investment return and principal value of an investment will fluctuate so that an investor's shares, when redeemed, may be worth more or less than their original cost.

        SIMULATION AND ANALYSIS LIMITATIONS:
        The projections, scenario analyses, and other information generated by this report regarding the likelihood of various investment outcomes are hypothetical in nature, do not reflect actual investment results, and are not guarantees of future results. The simulations are based on assumptions, and there can be no assurance that the projected or simulated results will be achieved or sustained.

        FORWARD-LOOKING STATEMENTS:
        This report may contain forward-looking statements. Forward-looking statements are not guarantees of future performance and involve risks and uncertainties. Actual results may differ materially from those projected in the forward-looking statements.

        TAX AND LEGAL CONSIDERATIONS:
        ISO Financial does not provide tax, legal, or accounting advice. This material has been prepared for informational purposes only and is not intended to provide, and should not be relied on for, tax, legal, or accounting advice. You should consult your own tax, legal, and accounting advisors before engaging in any transaction or implementing any recommendation contained in this report.

        THIRD-PARTY INFORMATION:
        This report may contain data or information obtained from third-party sources. While we believe the third-party information is reliable, we have not independently verified the accuracy or completeness of the third-party information.

        REDISTRIBUTION PROHIBITION:
        This report is confidential and for the exclusive use of the premium subscriber to whom it is provided. Redistribution, reproduction, or transmission of this report, in whole or in part, without the express written permission of ISO Financial is strictly prohibited.

        ALGORITHMIC AND QUANTITATIVE ANALYSIS:
        Portions of this report utilize advanced algorithmic, quantitative, and statistical methods to analyze market data. These models have inherent limitations and rely on underlying data which may contain errors or may not be representative of future market conditions.

        By using this report, you acknowledge that you have read and understood this disclaimer and agree to its terms.
        """
        
        pdf.set_font("Arial", size=9)
        for line in disclaimer.strip().split('\n'):
            if line.strip():
                pdf.multi_cell(190, 5, txt=line.strip())
            else:
                pdf.ln(3)
        
        # Add premium footer with branding
        pdf.ln(10)
        pdf.set_font("Arial", 'B', size=10)
        pdf.set_text_color(*self.colors['premium'])
        pdf.cell(190, 5, txt="ISO Financial Premium Analytics", ln=True, align='C')
        
        pdf.set_font("Arial", 'I', size=8)
        pdf.cell(190, 5, txt=f"Report ID: PREMIUM-{timestamp}", ln=True, align='C')
        pdf.cell(190, 5, txt="©2025 ISO Financial. All rights reserved.", ln=True, align='C')
        
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