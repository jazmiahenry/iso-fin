import os
import tempfile
from typing import Dict, Any, List, Optional, Union
import logging
import json
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import base64
from io import BytesIO
from fpdf import FPDF
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import qrcode

# Setup logger
logger = logging.getLogger(__name__)

class PDFReport:
    def __init__(self, report_type: str = "free"):
        """
        Initialize the PDF report generator
        
        Args:
            report_type: Type of report ("free" or "premium")
        """
        self.report_type = report_type  # 'free' or 'premium'
        
        # Configure PDF properties based on report type
        if report_type == "premium":
            self.pdf = FPDF(orientation='P', unit='mm', format='A4')
            self.pdf.set_author("ISO Financial")
            self.pdf.set_creator("ISO Financial Premium Report")
            self.pdf.set_title("ISO Financial Premium Portfolio Analysis")
            self.pdf.set_subject("Financial Portfolio Analysis")
        else:
            self.pdf = FPDF(orientation='P', unit='mm', format='A4')
            self.pdf.set_author("ISO Financial")
            self.pdf.set_creator("ISO Financial Free Report")
            self.pdf.set_title("ISO Financial Basic Portfolio Analysis")
            self.pdf.set_subject("Financial Portfolio Analysis")
        
        # Set default margins
        self.pdf.set_margins(left=15, top=15, right=15)
        
        # Skip custom fonts entirely - use only standard fonts
        # This avoids any font loading errors that might occur
        try:
            # Try standard Arial font first
            self.pdf.set_font('Arial', '', 10)
            self.font_family = 'Arial'
            logger.info("Using standard Arial font")
        except Exception as e:
            # If Arial fails, try Helvetica (built into FPDF)
            try:
                self.pdf.set_font('Helvetica', '', 10)
                self.font_family = 'Helvetica'
                logger.info("Using standard Helvetica font")
            except Exception as e2:
                # If both fail, try Times as last resort
                try:
                    self.pdf.set_font('Times', '', 10)
                    self.font_family = 'Times'
                    logger.info("Using standard Times font")
                except Exception as e3:
                    # If all fail, let FPDF use its built-in default
                    logger.warning(f"All font attempts failed: {str(e3)}, using FPDF default")
                    self.font_family = ''
        
        # Set flag for custom font availability - always false in this implementation
        self.custom_fonts_loaded = False
        
        # Track temporary files to clean up
        self.temp_files = []
    
    def __del__(self):
        """Clean up temporary files when object is destroyed"""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {temp_file}: {str(e)}")
                
    def _set_font(self, style='', size=10, family=None):
        """Helper method to set font with fallback to standard fonts"""
        try:
            # Use specified family, or our stored default, or Arial if nothing else
            font_family = family or getattr(self, 'font_family', '') or 'Arial'
            if font_family:
                self.pdf.set_font(font_family, style, size)
            else:
                # If no font family is set, let FPDF use its default with just style and size
                self.pdf.set_font(style=style, size=size)
        except Exception as e:
            # Emergency fallback if the first attempt fails
            logger.warning(f"Font error: {str(e)}, trying fallbacks")
            
            # Try Helvetica (built into FPDF)
            try:
                self.pdf.set_font('Helvetica', style, size)
            except Exception:
                # Try Times as last resort
                try:
                    self.pdf.set_font('Times', style, size)
                except Exception:
                    # If all else fails, try to just set the style and size
                    try:
                        self.pdf.set_font(style=style, size=size)
                    except Exception:
                        # Complete failure - log but continue without changing font
                        logger.error("All font attempts failed, continuing with current font")
                        pass
    
    def generate_report(self, 
                      query_data: Dict[str, Any], 
                      simulation_results: Dict[str, Any],
                      report_data: Dict[str, Any],
                      output_file: Optional[str] = None) -> str:
        """
        Generate a PDF report
        
        Args:
            query_data: Original query data
            simulation_results: Simulation results
            report_data: Formatted report data
            output_file: Optional output file path
            
        Returns:
            Path to the generated PDF file
        """
        try:
            # Add report header
            self._add_header()
            
            # Add user query summary
            self._add_query_summary(query_data)
            
            # Add appropriate report content based on type
            if self.report_type == "premium":
                # Premium report includes all content
                self._add_portfolio_overview(query_data, report_data)
                self._add_monte_carlo_simulation(simulation_results.get("monte_carlo", {}))
                self._add_risk_analysis(simulation_results.get("cvar", {}), simulation_results.get("risk_parity", {}))
                self._add_scenario_analysis(simulation_results.get("scenario", {}))
                self._add_recommendations(report_data.get("recommendations", []))
                self._add_disclaimer(premium=True)
            else:
                # Free report includes limited content
                self._add_portfolio_overview(query_data, report_data, basic=True)
                self._add_monte_carlo_simulation(simulation_results.get("monte_carlo", {}), basic=True)
                self._add_basic_risk_overview(simulation_results.get("monte_carlo", {}))
                self._add_disclaimer()
                self._add_premium_upsell()
            
            # Set output file path
            if output_file is None:
                timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                if self.report_type == "premium":
                    output_file = f"iso_financial_premium_{timestamp}.pdf"
                else:
                    output_file = f"iso_financial_report_{timestamp}.pdf"
            
            # Output PDF
            self.pdf.output(output_file)
            logger.info(f"Generated PDF report: {output_file}")
            
            return output_file
        
        except Exception as e:
            logger.error(f"Error generating PDF report: {str(e)}")
            raise
    
    def _add_header(self):
        """Add report header"""
        self.pdf.add_page()
        
        # Try to add logo from new location
        try:
            # Check the provided logo path first
            logo_path = os.path.join(os.path.dirname(__file__), 'img', 'isoai_symbol.png')
            
            # More verbose debugging
            logger.info(f"Checking for logo at primary path: {logo_path}")
            
            # Fallback paths to check
            fallback_paths = [
                os.path.join(os.path.dirname(__file__), 'assets', 'iso_logo.png'),
                os.path.join(os.path.dirname(__file__), 'assets', 'logo.png'),
                os.path.join(os.path.dirname(__file__), 'assets', 'isoai_symbol.png'),
                '/Users/jazmiahenry/reports/iso_financial_mvp/report_generator/img/isoai_symbol.png'  # Absolute path backup
            ]
            
            for path in fallback_paths:
                logger.info(f"Fallback path to check: {path} (exists: {os.path.exists(path)})")
            
            # Try the main logo path first
            if os.path.exists(logo_path):
                self.pdf.image(logo_path, x=15, y=15, w=40)
                logger.info(f"Using logo from: {logo_path}")
            else:
                # Try fallback paths
                logo_found = False
                for path in fallback_paths:
                    if os.path.exists(path):
                        try:
                            self.pdf.image(path, x=15, y=15, w=40)
                            logger.info(f"Using fallback logo from: {path}")
                            logo_found = True
                            break
                        except Exception as img_err:
                            logger.warning(f"Error using logo at {path}: {str(img_err)}")
                
                if not logo_found:
                    # Skip logo and log a message
                    logger.warning(f"Logo file not found at {logo_path} or any fallback locations")
        except Exception as e:
            # Continue without logo if there's an error
            logger.warning(f"Error adding logo: {str(e)}")
        
        # Add title (with or without logo)
        self.pdf.set_xy(15, 30)
        
        # Set title font
        self._set_font('B', 20)
        
        if self.report_type == "premium":
            self.pdf.set_text_color(31, 64, 122)  # Dark blue for premium
            self.pdf.cell(180, 10, "Iso AI Founding Access Report", 0, 1, 'C')
        else:
            self.pdf.set_text_color(80, 80, 80)  # Dark gray for free
            self.pdf.cell(180, 10, "Iso AI Financial Report", 0, 1, 'C')
        
        # Add date
        self._set_font('', 10)
        self.pdf.set_text_color(100, 100, 100)
        current_date = datetime.datetime.now().strftime("%B %d, %Y")
        self.pdf.cell(180, 10, f"Generated on {current_date}", 0, 1, 'C')
        
        # Add separator line
        self.pdf.set_draw_color(200, 200, 200)
        self.pdf.line(15, 55, 195, 55)
        
        # Reset text color
        self.pdf.set_text_color(0, 0, 0)
    
    def _add_query_summary(self, query_data: Dict[str, Any]):
        """Add user query summary"""
        self.pdf.set_xy(15, 60)
        
        # Add section title
        self._set_font('B', 14)
        self.pdf.cell(180, 10, "Your Financial Query", 0, 1)
        
        # Add original query
        self._set_font('B', 12)
        original_query = query_data.get("original_query", "No query provided")
        
        # Handle multi-line query
        self._set_font('', 11)
        self.pdf.set_text_color(50, 50, 50)
        self.pdf.multi_cell(180, 7, f'"{original_query}"', 0, 'L')
        
        # Add query details
        self.pdf.ln(5)
        self._set_font('B', 11)
        self.pdf.set_text_color(0, 0, 0)
        self.pdf.cell(180, 7, "Query Details:", 0, 1)
        
        self._set_font('', 10)
        
        # Query type
        query_type = query_data.get("query_type", "portfolio_analysis")
        query_type_formatted = query_type.replace("_", " ").title()
        self.pdf.cell(60, 7, "Analysis Type:", 0, 0)
        self.pdf.cell(120, 7, query_type_formatted, 0, 1)
        
        # Timeframe
        timeframe = query_data.get("timeframe", {})
        timeframe_value = timeframe.get("value", 10)
        timeframe_unit = timeframe.get("unit", "years")
        self.pdf.cell(60, 7, "Time Horizon:", 0, 0)
        self.pdf.cell(120, 7, f"{timeframe_value} {timeframe_unit}", 0, 1)
        
        # Risk profile
        risk_profile = query_data.get("risk_profile", "moderate")
        self.pdf.cell(60, 7, "Risk Profile:", 0, 0)
        self.pdf.cell(120, 7, risk_profile.capitalize(), 0, 1)
        
        # Events (if present)
        events = query_data.get("events", [])
        if events:
            self.pdf.cell(60, 7, "Scenario:", 0, 0)
            event_desc = self._format_event_description(events[0])
            self.pdf.cell(120, 7, event_desc, 0, 1)
        
        # Add separator line
        self.pdf.ln(3)
        self.pdf.set_draw_color(200, 200, 200)
        self.pdf.line(15, self.pdf.get_y(), 195, self.pdf.get_y())
        self.pdf.ln(5)
    
    def _format_event_description(self, event: Dict[str, Any]) -> str:
        """Format event description for display"""
        event_type = event.get("type", "")
        
        if event_type == "price_change":
            asset = event.get("asset", "asset")
            change = event.get("change", 0)
            return f"{asset.title()} price change of {change * 100:.0f}%"
        
        elif event_type == "market_crash":
            severity = event.get("severity", "medium")
            return f"Market crash (severity: {severity})"
        
        elif event_type == "regulatory_event":
            return "Regulatory event"
        
        else:
            return "Custom scenario"
    
    def _add_portfolio_overview(self, query_data: Dict[str, Any], report_data: Dict[str, Any], basic: bool = False):
        """Add portfolio overview section"""
        # Add section title
        self._set_font('B', 14)
        self.pdf.cell(180, 10, "Portfolio Overview", 0, 1)
        
        # Get assets from query data
        assets = query_data.get("assets", [])
        
        if not assets and "portfolio" in query_data:
            assets = query_data.get("portfolio", [])
        
        if not assets:
            self._set_font('', 10)
            self.pdf.cell(180, 7, "No portfolio data available", 0, 1)
            return
        
        # Create assets table
        self._set_font('B', 10)
        self.pdf.cell(60, 7, "Asset Type", 1, 0, 'C')
        self.pdf.cell(60, 7, "Asset Name", 1, 0, 'C')
        self.pdf.cell(30, 7, "Weight", 1, 0, 'C')
        
        if not basic and self.report_type == "premium":
            self.pdf.cell(30, 7, "Risk", 1, 1, 'C')
        else:
            self.pdf.cell(30, 7, "", 1, 1, 'C')
        
        # Add assets
        self._set_font('', 10)
        
        for asset in assets:
            asset_type = asset.get("type", "")
            asset_name = asset.get("name", "")
            weight = asset.get("weight", 0)
            
            self.pdf.cell(60, 7, asset_type.capitalize(), 1, 0)
            self.pdf.cell(60, 7, asset_name.replace("_", " ").title(), 1, 0)
            self.pdf.cell(30, 7, f"{weight * 100:.1f}%", 1, 0, 'R')
            
            if not basic and self.report_type == "premium":
                # Add risk for premium reports
                risk_level = "Medium"
                if asset_type == "crypto" or asset_type == "nft":
                    risk_level = "High"
                elif asset_type == "bonds" or asset_type == "cash":
                    risk_level = "Low"
                
                self.pdf.cell(30, 7, risk_level, 1, 1, 'C')
            else:
                self.pdf.cell(30, 7, "", 1, 1)
        
        # Add asset allocation pie chart
        if len(assets) > 0:
            pie_chart_path = self._create_asset_allocation_chart(assets)
            if pie_chart_path:
                self.pdf.ln(5)
                self.pdf.image(pie_chart_path, x=55, y=None, w=100)
                self.pdf.ln(60)  # Space for the image
        
        # Add separator
        self.pdf.ln(5)
        self.pdf.set_draw_color(200, 200, 200)
        self.pdf.line(15, self.pdf.get_y(), 195, self.pdf.get_y())
        self.pdf.ln(5)
    
    def _create_asset_allocation_chart(self, assets: List[Dict[str, Any]]) -> Optional[str]:
        """Create asset allocation pie chart"""
        try:
            # Extract data
            labels = [asset.get("name", "").replace("_", " ").title() for asset in assets]
            sizes = [asset.get("weight", 0) for asset in assets]
            
            # Create color map based on asset type
            colors = []
            for asset in assets:
                asset_type = asset.get("type", "")
                if asset_type == "crypto":
                    colors.append("#5470C6")  # Blue
                elif asset_type == "nft":
                    colors.append("#91CC75")  # Green
                elif asset_type == "traditional":
                    colors.append("#FAC858")  # Yellow
                elif asset_type == "alternative":
                    colors.append("#EE6666")  # Red
                elif asset_type == "real_estate":
                    colors.append("#73C0DE")  # Light blue
                elif asset_type == "collectible":
                    colors.append("#3BA272")  # Teal
                else:
                    colors.append("#FC8452")  # Orange
            
            # Create pie chart
            plt.figure(figsize=(10, 6))
            plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            plt.axis('equal')
            plt.title('Portfolio Asset Allocation')
            
            # Save to temporary file
            fd, temp_path = tempfile.mkstemp(suffix='.png')
            plt.savefig(temp_path, bbox_inches='tight')
            plt.close()
            os.close(fd)
            
            # Add to list of temporary files for cleanup
            self.temp_files.append(temp_path)
            
            return temp_path
        
        except Exception as e:
            logger.error(f"Error creating asset allocation chart: {str(e)}")
            return None
    
    def _add_monte_carlo_simulation(self, mc_results: Dict[str, Any], basic: bool = False):
        """Add Monte Carlo simulation section"""
        # We may not have simulation_paths in our test data, so let's be more lenient
        if not mc_results:
            return
        
        # Add section title
        self._set_font('B', 14)
        self.pdf.cell(180, 10, "Portfolio Projection", 0, 1)
        
        # Add brief description
        self._set_font('', 10)
        
        description = (
            "This forecast shows how your portfolio may perform over time. "
            "The lines represent different possible scenarios based on historical returns and volatility."
        )
        
        self.pdf.multi_cell(180, 7, description, 0, 'L')
        self.pdf.ln(3)
        
        # Create and add chart
        chart_path = self._create_monte_carlo_chart(mc_results, basic)
        if chart_path:
            self.pdf.image(chart_path, x=25, y=None, w=160)
            self.pdf.ln(80)  # Space for the image
        
        # Add statistics - if not present in statistics, try to get from the main monte carlo results
        stats = mc_results.get("statistics", {})
        
        # If we don't have statistics but have final values, create them
        if not stats and "mean_final" in mc_results and "median_final" in mc_results:
            stats = {
                "mean": mc_results.get("mean_final", 0),
                "median": mc_results.get("median_final", 0)
            }
            
            # If we have percentiles, add them as confidence intervals
            percentiles = mc_results.get("percentiles", {})
            if percentiles and 5 in percentiles and 95 in percentiles:
                stats["confidence_intervals"] = {
                    "p5": percentiles[5],
                    "p95": percentiles[95]
                }
        
        if stats:
            self._set_font('B', 12)
            self.pdf.cell(180, 10, "Projection Statistics", 0, 1)
            
            self._set_font('', 10)
            
            # Create two-column layout for statistics
            col_width = 85
            
            # Mean and median - try different key patterns that might exist
            mean_value = stats.get('mean', stats.get('mean_final', 0))
            median_value = stats.get('median', stats.get('median_final', 0))
            
            self._set_font('B', 10)
            self.pdf.cell(col_width, 7, "Mean Final Value:", 0, 0)
            self._set_font('', 10)
            self.pdf.cell(col_width, 7, f"${mean_value:,.2f}", 0, 1)
            
            self._set_font('B', 10)
            self.pdf.cell(col_width, 7, "Median Final Value:", 0, 0)
            self._set_font('', 10)
            self.pdf.cell(col_width, 7, f"${median_value:,.2f}", 0, 1)
            
            # 95% confidence interval
            confidence_intervals = stats.get("confidence_intervals", {})
            
            # Add percentiles if available
            if "percentiles" in mc_results and not confidence_intervals:
                percentiles = mc_results.get("percentiles", {})
                if percentiles and 5 in percentiles and 95 in percentiles:
                    confidence_intervals = {
                        "p5": percentiles[5],
                        "p95": percentiles[95]
                    }
            
            if confidence_intervals and "p5" in confidence_intervals and "p95" in confidence_intervals:
                self._set_font('B', 10)
                self.pdf.cell(col_width, 7, "90% Confidence Range:", 0, 0)
                self._set_font('', 10)
                lower = confidence_intervals.get("p5", 0)
                upper = confidence_intervals.get("p95", 0)
                self.pdf.cell(col_width, 7, f"${lower:,.2f} to ${upper:,.2f}", 0, 1)
        
        # Add separator
        self.pdf.ln(5)
        self.pdf.set_draw_color(200, 200, 200)
        self.pdf.line(15, self.pdf.get_y(), 195, self.pdf.get_y())
        self.pdf.ln(5)
    
    def _create_monte_carlo_chart(self, mc_results: Dict[str, Any], basic: bool = False) -> Optional[str]:
        """Create Monte Carlo simulation chart"""
        try:
            paths = mc_results.get("simulation_paths")
            
            # If we don't have simulation paths, create a simple mock chart
            if paths is None:
                logger.info("No simulation paths found, creating simple mock chart")
                # Create a mock chart for demo/testing purposes
                plt.figure(figsize=(12, 7))
                
                # Initial investment
                initial = mc_results.get("initial_investment", 100000)
                
                # Generate some simple mock paths (just for visualization purposes)
                num_steps = 120  # 10 years with monthly steps
                num_paths = 3    # Just a few paths for the mock
                
                # Mock paths with some reasonable returns
                mock_paths = []
                for i in range(num_paths):
                    # Different growth rates for different percentiles
                    if i == 0:  # Lower 5th percentile
                        annual_return = 0.02  # 2% annual return for pessimistic scenario 
                    elif i == 1:  # Median
                        annual_return = 0.07  # 7% annual return for median scenario
                    else:  # Upper 95th percentile
                        annual_return = 0.12  # 12% annual return for optimistic scenario
                        
                    # Convert annual to monthly return
                    monthly_return = (1 + annual_return) ** (1/12) - 1
                    
                    # Generate path with some random noise
                    path = [initial]
                    for m in range(1, num_steps):
                        noise = np.random.normal(0, 0.02)  # Some random noise
                        path.append(path[-1] * (1 + monthly_return + noise))
                    
                    mock_paths.append(path)
                
                # Convert to numpy array
                paths = np.array(mock_paths)
                
                # Add note about mock data to the chart
                plt.figtext(0.5, 0.01, "Note: Using simulated data for visualization purposes", 
                           ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.2})
            
            # Convert to numpy array if it's a list
            elif isinstance(paths, list):
                paths = np.array(paths)
            
            # Get number of time steps
            if len(paths.shape) < 2:
                logger.warning("Paths data is not in expected format")
                return None
                
            _, num_steps = paths.shape
            
            # Create figure
            plt.figure(figsize=(12, 7))
            
            # Limit the number of paths in the basic version
            if basic:
                # Plot just percentile lines in the basic version
                percentiles = [5, 50, 95]
                colors = ['red', 'black', 'green']
                labels = ['5th Percentile', 'Median', '95th Percentile']
                
                for p, color, label in zip(percentiles, colors, labels):
                    percentile_values = np.percentile(paths, p, axis=0)
                    plt.plot(percentile_values, color=color, linewidth=2, label=label)
            
            else:
                # Premium version - plot a sample of paths
                num_paths_to_plot = min(100, len(paths))
                path_indices = np.random.choice(len(paths), num_paths_to_plot, replace=False)
                
                for i in path_indices:
                    plt.plot(paths[i], color='steelblue', alpha=0.1, linewidth=0.5)
                
                # Add percentile lines
                percentiles = [5, 50, 95]
                colors = ['red', 'black', 'green']
                labels = ['5th Percentile', 'Median', '95th Percentile']
                
                for p, color, label in zip(percentiles, colors, labels):
                    percentile_values = np.percentile(paths, p, axis=0)
                    plt.plot(percentile_values, color=color, linewidth=2, label=label)
            
            # Set titles and labels
            plt.title('Portfolio Value Projection', fontsize=14)
            plt.xlabel('Time (Months)', fontsize=12)
            plt.ylabel('Portfolio Value ($)', fontsize=12)
            
            # Add gridlines
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Add legend
            plt.legend()
            
            # Create x-axis labels (years)
            years = num_steps // 12
            x_ticks = [i * 12 for i in range(years + 1)]
            x_labels = [f"{i}y" for i in range(years + 1)]
            plt.xticks(x_ticks, x_labels)
            
            # Add dollar formatting to y-axis
            plt.gca().yaxis.set_major_formatter('${x:,.0f}')
            
            # Save to temporary file
            fd, temp_path = tempfile.mkstemp(suffix='.png')
            plt.savefig(temp_path, bbox_inches='tight')
            plt.close()
            os.close(fd)
            
            # Add to list of temporary files for cleanup
            self.temp_files.append(temp_path)
            
            return temp_path
        
        except Exception as e:
            logger.error(f"Error creating Monte Carlo chart: {str(e)}")
            return None
    
    def _add_basic_risk_overview(self, mc_results: Dict[str, Any]):
        """Add basic risk overview for free reports"""
        if not mc_results or "statistics" not in mc_results:
            return
        
        # Add section title
        self._set_font('B', 14)
        self.pdf.cell(180, 10, "Risk Overview", 0, 1)
        
        # Add brief description
        self._set_font('', 10)
        
        description = (
            "This section provides a basic overview of the risk in your portfolio. "
            "Understanding these metrics can help you make more informed investment decisions."
        )
        
        self.pdf.multi_cell(180, 7, description, 0, 'L')
        self.pdf.ln(3)
        
        # Extract statistics
        stats = mc_results.get("statistics", {})
        
        if stats:
            # Create two-column layout for risk metrics
            col_width = 85
            
            # Value at Risk
            var_95 = stats.get("var_95", 0)
            initial_investment = mc_results.get("params", {}).get("initial_investment", 100000)
            
            self._set_font('B', 10)
            self.pdf.cell(col_width, 7, "Value at Risk (VaR, 95%):", 0, 0)
            self._set_font('', 10)
            self.pdf.cell(col_width, 7, f"${var_95:,.2f}", 0, 1)
            
            self._set_font('', 9)
            self.pdf.cell(180, 7, "This represents the minimum loss expected in 5% of scenarios.", 0, 1)
            self.pdf.ln(2)
            
            # Maximum Drawdown
            max_drawdown = stats.get("max_drawdown", 0)
            
            self._set_font('B', 10)
            self.pdf.cell(col_width, 7, "Maximum Drawdown:", 0, 0)
            self._set_font('', 10)
            self.pdf.cell(col_width, 7, f"{max_drawdown * 100:.1f}%", 0, 1)
            
            self._set_font('', 9)
            self.pdf.cell(180, 7, "This represents the maximum portfolio decline from peak to trough.", 0, 1)
            self.pdf.ln(2)
            
            # Static scenario (market crash)
            self._set_font('B', 10)
            self.pdf.cell(col_width, 7, "Market Crash Scenario:", 0, 0)
            
            # Simulate market crash effect (simplified)
            crash_impact = -0.3  # 30% loss
            crash_value = initial_investment * (1 + crash_impact)
            
            self._set_font('', 10)
            self.pdf.cell(col_width, 7, f"Portfolio value could decline to ${crash_value:,.2f}", 0, 1)
            
            self._set_font('', 9)
            self.pdf.cell(180, 7, "This represents portfolio value in a 30% market crash scenario.", 0, 1)
        
        # Add separator
        self.pdf.ln(5)
        self.pdf.set_draw_color(200, 200, 200)
        self.pdf.line(15, self.pdf.get_y(), 195, self.pdf.get_y())
        self.pdf.ln(5)
    
    def _add_risk_analysis(self, cvar_results: Dict[str, Any], risk_parity_results: Dict[str, Any]):
        """Add detailed risk analysis for premium reports"""
        # Add section title
        self._set_font('B', 14)
        self.pdf.cell(180, 10, "Risk Analysis", 0, 1)
        
        # Add VaR and CVaR analysis
        if cvar_results:
            self._set_font('B', 12)
            self.pdf.cell(180, 10, "Tail Risk Metrics", 0, 1)
            
            # Add brief description
            self._set_font('', 10)
            
            description = (
                "These metrics quantify potential losses in adverse scenarios. "
                "Value at Risk (VaR) represents the minimum loss in the worst 5% of scenarios, "
                "while Conditional VaR represents the average loss in those worst scenarios."
            )
            
            self.pdf.multi_cell(180, 7, description, 0, 'L')
            self.pdf.ln(3)
            
            # Create table for VaR metrics
            self._set_font('B', 10)
            self.pdf.cell(90, 7, "Risk Metric", 1, 0, 'C')
            self.pdf.cell(45, 7, "Value ($)", 1, 0, 'C')
            self.pdf.cell(45, 7, "% of Portfolio", 1, 1, 'C')
            
            self._set_font('', 10)
            
            # Add VaR rows
            var_amount = cvar_results.get("var_amount", 0)
            var_pct = cvar_results.get("var", 0)
            self.pdf.cell(90, 7, "Value at Risk (95% confidence)", 1, 0)
            self.pdf.cell(45, 7, f"${var_amount:,.2f}", 1, 0, 'R')
            self.pdf.cell(45, 7, f"{var_pct * 100:.1f}%", 1, 1, 'R')
            
            # Add CVaR rows
            cvar_amount = cvar_results.get("cvar_amount", 0)
            cvar_pct = cvar_results.get("cvar", 0)
            self.pdf.cell(90, 7, "Conditional VaR (95% confidence)", 1, 0)
            self.pdf.cell(45, 7, f"${cvar_amount:,.2f}", 1, 0, 'R')
            self.pdf.cell(45, 7, f"{cvar_pct * 100:.1f}%", 1, 1, 'R')
            
            # Add worst loss
            worst_loss = cvar_results.get("worst_loss", 0)
            worst_loss_pct = worst_loss / cvar_results.get("initial_investment", 100000) if "initial_investment" in cvar_results else 0
            self.pdf.cell(90, 7, "Worst Case Loss", 1, 0)
            self.pdf.cell(45, 7, f"${worst_loss:,.2f}", 1, 0, 'R')
            self.pdf.cell(45, 7, f"{worst_loss_pct * 100:.1f}%", 1, 1, 'R')
            
            self.pdf.ln(5)
        
        # Add risk contribution analysis
        if risk_parity_results and "asset_details" in risk_parity_results:
            self._set_font('B', 12)
            self.pdf.cell(180, 10, "Risk Contribution by Asset", 0, 1)
            
            # Add brief description
            self._set_font('', 10)
            
            description = (
                "This analysis shows how each asset contributes to the overall portfolio risk. "
                "A well-diversified portfolio typically distributes risk more evenly across assets."
            )
            
            self.pdf.multi_cell(180, 7, description, 0, 'L')
            self.pdf.ln(3)
            
            # Create risk contribution chart
            chart_path = self._create_risk_contribution_chart(risk_parity_results)
            if chart_path:
                self.pdf.image(chart_path, x=35, y=None, w=140)
                self.pdf.ln(80)  # Space for the image
            
            # Add optimal allocation suggestion (if available)
            if "trades" in risk_parity_results:
                self._set_font('B', 12)
                self.pdf.cell(180, 10, "Suggested Risk-Balanced Allocation", 0, 1)
                
                # Add brief description
                self._set_font('', 10)
                
                description = (
                    "This suggestion shows a risk-parity optimized allocation that may provide "
                    "more balanced risk exposure across your portfolio assets."
                )
                
                self.pdf.multi_cell(180, 7, description, 0, 'L')
                self.pdf.ln(3)
                
                # Add table for suggested trades
                trades = risk_parity_results.get("trades", {}).get("trades", [])
                
                if trades:
                    self._set_font('B', 10)
                    self.pdf.cell(60, 7, "Asset Name", 1, 0, 'C')
                    self.pdf.cell(30, 7, "Action", 1, 0, 'C')
                    self.pdf.cell(45, 7, "Amount ($)", 1, 0, 'C')
                    self.pdf.cell(45, 7, "New Weight", 1, 1, 'C')
                    
                    self._set_font('', 10)
                    
                    for trade in trades:
                        asset_name = trade.get("asset_name", "").replace("_", " ").title()
                        direction = trade.get("direction", "hold").capitalize()
                        amount = trade.get("amount", 0)
                        target_weight = trade.get("target_weight", 0)
                        
                        self.pdf.cell(60, 7, asset_name, 1, 0)
                        self.pdf.cell(30, 7, direction, 1, 0, 'C')
                        self.pdf.cell(45, 7, f"${amount:,.2f}", 1, 0, 'R')
                        self.pdf.cell(45, 7, f"{target_weight * 100:.1f}%", 1, 1, 'R')
                    
                    # Add risk reduction estimate
                    risk_reduction = risk_parity_results.get("trades", {}).get("estimated_risk_reduction", 0)
                    
                    self.pdf.ln(3)
                    self._set_font('B', 10)
                    self.pdf.cell(120, 7, "Estimated Risk Reduction:", 0, 0)
                    self._set_font('', 10)
                    self.pdf.cell(60, 7, f"{risk_reduction * 100:.1f}%", 0, 1)
        
        # Add separator
        self.pdf.ln(5)
        self.pdf.set_draw_color(200, 200, 200)
        self.pdf.line(15, self.pdf.get_y(), 195, self.pdf.get_y())
        self.pdf.ln(5)
    
    def _create_risk_contribution_chart(self, risk_parity_results: Dict[str, Any]) -> Optional[str]:
        """Create risk contribution pie chart"""
        try:
            asset_details = risk_parity_results.get("asset_details", [])
            
            if not asset_details:
                return None
            
            # Extract data
            labels = [detail.get("name", "").replace("_", " ").title() for detail in asset_details]
            sizes = [detail.get("risk_contribution_pct", 0) for detail in asset_details]
            
            # Create pie chart
            plt.figure(figsize=(10, 6))
            plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            plt.axis('equal')
            plt.title('Risk Contribution by Asset')
            
            # Save to temporary file
            fd, temp_path = tempfile.mkstemp(suffix='.png')
            plt.savefig(temp_path, bbox_inches='tight')
            plt.close()
            os.close(fd)
            
            # Add to list of temporary files for cleanup
            self.temp_files.append(temp_path)
            
            return temp_path
        
        except Exception as e:
            logger.error(f"Error creating risk contribution chart: {str(e)}")
            return None
    
    def _add_scenario_analysis(self, scenario_results: Dict[str, Any]):
        """Add scenario analysis for premium reports"""
        if not scenario_results or "comparison" not in scenario_results:
            return
        
        # Check if we're on a new page or need to add one
        if self.pdf.get_y() > 200:
            self.pdf.add_page()
        
        # Add section title
        self._set_font('B', 14)
        self.pdf.cell(180, 10, "Scenario Analysis", 0, 1)
        
        # Get scenario details
        scenario_details = scenario_results.get("scenario_details", {})
        scenario_name = scenario_details.get("name", "Custom Scenario")
        scenario_description = scenario_details.get("description", "")
        
        # Add scenario description
        self._set_font('B', 12)
        self.pdf.cell(180, 10, f"Scenario: {scenario_name}", 0, 1)
        
        if scenario_description:
            self._set_font('', 10)
            self.pdf.multi_cell(180, 7, scenario_description, 0, 'L')
            self.pdf.ln(3)
        
        # Add impact analysis
        comparison = scenario_results.get("comparison", {})
        
        if comparison:
            # Extract key metrics
            mean_change = comparison.get("mean_change", {}).get("percentage", 0)
            var_change = comparison.get("risk_measures", {}).get("var_change", {}).get("percentage", 0)
            
            # Add projected impact
            self._set_font('B', 11)
            self.pdf.cell(180, 10, "Projected Impact", 0, 1)
            
            self._set_font('', 10)
            impact_text = f"In this scenario, your portfolio's projected return could change by {mean_change * 100:.1f}% "
            impact_text += f"and your Value at Risk (VaR) could change by {var_change * 100:.1f}%."
            
            self.pdf.multi_cell(180, 7, impact_text, 0, 'L')
            self.pdf.ln(3)
            
            # Create scenario comparison chart
            chart_path = self._create_scenario_comparison_chart(scenario_results)
            if chart_path:
                self.pdf.image(chart_path, x=35, y=None, w=140)
                self.pdf.ln(80)  # Space for the image
            
            # Add recovery analysis
            recovery = comparison.get("recovery_analysis", {})
            
            if recovery:
                self._set_font('B', 11)
                self.pdf.cell(180, 10, "Recovery Analysis", 0, 1)
                
                # Extract recovery metrics
                paths_recover = recovery.get("paths_that_recover_pct", 0)
                avg_recovery_time = recovery.get("avg_recovery_time_months", 0)
                
                self._set_font('', 10)
                recovery_text = f"In this scenario, {paths_recover * 100:.1f}% of simulated paths "
                recovery_text += f"recover to their initial value, with an average recovery time of {avg_recovery_time:.1f} months."
                
                self.pdf.multi_cell(180, 7, recovery_text, 0, 'L')
                
                # Add probability of loss table
                prob_loss = comparison.get("probability_of_loss", {})
                
                if prob_loss:
                    self.pdf.ln(3)
                    self._set_font('B', 10)
                    self.pdf.cell(60, 7, "Loss Threshold", 1, 0, 'C')
                    self.pdf.cell(40, 7, "Baseline", 1, 0, 'C')
                    self.pdf.cell(40, 7, "Scenario", 1, 0, 'C')
                    self.pdf.cell(40, 7, "Difference", 1, 1, 'C')
                    
                    self._set_font('', 10)
                    
                    # Any loss
                    prob_any = prob_loss.get("probability_any_loss", {})
                    self.pdf.cell(60, 7, "Any Loss", 1, 0)
                    self.pdf.cell(40, 7, f"{prob_any.get('baseline', 0) * 100:.1f}%", 1, 0, 'R')
                    self.pdf.cell(40, 7, f"{prob_any.get('scenario', 0) * 100:.1f}%", 1, 0, 'R')
                    self.pdf.cell(40, 7, f"{prob_any.get('difference', 0) * 100:.1f}%", 1, 1, 'R')
                    
                    # 10% loss
                    prob_10 = prob_loss.get("probability_10pct_loss", {})
                    self.pdf.cell(60, 7, "10% Loss", 1, 0)
                    self.pdf.cell(40, 7, f"{prob_10.get('baseline', 0) * 100:.1f}%", 1, 0, 'R')
                    self.pdf.cell(40, 7, f"{prob_10.get('scenario', 0) * 100:.1f}%", 1, 0, 'R')
                    self.pdf.cell(40, 7, f"{prob_10.get('difference', 0) * 100:.1f}%", 1, 1, 'R')
                    
                    # 25% loss
                    prob_25 = prob_loss.get("probability_25pct_loss", {})
                    self.pdf.cell(60, 7, "25% Loss", 1, 0)
                    self.pdf.cell(40, 7, f"{prob_25.get('baseline', 0) * 100:.1f}%", 1, 0, 'R')
                    self.pdf.cell(40, 7, f"{prob_25.get('scenario', 0) * 100:.1f}%", 1, 0, 'R')
                    self.pdf.cell(40, 7, f"{prob_25.get('difference', 0) * 100:.1f}%", 1, 1, 'R')
        
        # Add separator
        self.pdf.ln(5)
        self.pdf.set_draw_color(200, 200, 200)
        self.pdf.line(15, self.pdf.get_y(), 195, self.pdf.get_y())
        self.pdf.ln(5)
    
    def _create_scenario_comparison_chart(self, scenario_results: Dict[str, Any]) -> Optional[str]:
        """Create scenario comparison chart"""
        try:
            comparison = scenario_results.get("comparison", {})
            scenario_name = scenario_results.get("scenario_details", {}).get("name", "Scenario")
            
            if not comparison:
                return None
            
            # Extract key metrics for comparison
            metrics = [
                "Mean Return",
                "Median Return",
                "Value at Risk",
                "Conditional VaR",
                "Max Drawdown"
            ]
            
            # Extract values (as percentages)
            scenario_values = [
                comparison.get("mean_change", {}).get("percentage", 0) * 100,
                comparison.get("median_change", {}).get("percentage", 0) * 100,
                comparison.get("risk_measures", {}).get("var_change", {}).get("percentage", 0) * 100,
                comparison.get("risk_measures", {}).get("cvar_change", {}).get("percentage", 0) * 100,
                comparison.get("risk_measures", {}).get("drawdown_change", {}).get("percentage", 0) * 100
            ]
            
            # Create bar chart
            plt.figure(figsize=(12, 7))
            bar_colors = ['green' if val >= 0 else 'red' for val in scenario_values]
            
            plt.bar(metrics, scenario_values, color=bar_colors)
            
            # Add horizontal line at 0
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            # Set titles and labels
            plt.title(f'Impact of {scenario_name} Scenario', fontsize=14)
            plt.ylabel('Change (%)', fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Add value labels on bars
            for i, v in enumerate(scenario_values):
                sign = "+" if v >= 0 else ""
                plt.text(i, v + (1 if v >= 0 else -1), f"{sign}{v:.1f}%", 
                        ha='center', va='bottom' if v >= 0 else 'top')
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=15)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save to temporary file
            fd, temp_path = tempfile.mkstemp(suffix='.png')
            plt.savefig(temp_path, bbox_inches='tight')
            plt.close()
            os.close(fd)
            
            # Add to list of temporary files for cleanup
            self.temp_files.append(temp_path)
            
            return temp_path
        
        except Exception as e:
            logger.error(f"Error creating scenario comparison chart: {str(e)}")
            return None
    
    def _add_recommendations(self, recommendations: List[Dict[str, Any]]):
        """Add recommendations section for premium reports"""
        if not recommendations:
            return
        
        # Check if we're on a new page or need to add one
        if self.pdf.get_y() > 220:
            self.pdf.add_page()
        
        # Add section title
        self._set_font('B', 14)
        self.pdf.cell(180, 10, "Recommendations", 0, 1)
        
        # Add brief description
        self._set_font('', 10)
        
        description = (
            "Based on our analysis, we offer the following recommendations for your portfolio. "
            "These are suggestions to consider as part of your overall investment strategy."
        )
        
        self.pdf.multi_cell(180, 7, description, 0, 'L')
        self.pdf.ln(3)
        
        # Add each recommendation
        for i, recommendation in enumerate(recommendations, 1):
            title = recommendation.get("title", f"Recommendation {i}")
            rec_description = recommendation.get("description", "")
            
            self._set_font('B', 11)
            self.pdf.cell(180, 7, f"{i}. {title}", 0, 1)
            
            if rec_description:
                self._set_font('', 10)
                self.pdf.multi_cell(180, 7, rec_description, 0, 'L')
            
            self.pdf.ln(3)
        
        # Add separator
        self.pdf.ln(5)
        self.pdf.set_draw_color(200, 200, 200)
        self.pdf.line(15, self.pdf.get_y(), 195, self.pdf.get_y())
        self.pdf.ln(5)
    
    def _add_disclaimer(self, premium: bool = False):
        """Add disclaimer"""
        # Check if we're on a new page or need to add one
        if self.pdf.get_y() > 230:
            self.pdf.add_page()
        
        # Add section title
        self._set_font('B', 12)
        self.pdf.cell(180, 10, "Disclaimer", 0, 1)
        
        # Add disclaimer text
        self._set_font('', 8)
        
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
        
        self.pdf.multi_cell(180, 5, disclaimer_text, 0, 'L')
        
        # Add generation date
        self.pdf.ln(3)
        self._set_font('', 8)
        current_date = datetime.datetime.now().strftime("%B %d, %Y at %H:%M:%S")
        self.pdf.cell(180, 5, f"Generated on {current_date}", 0, 1, 'R')
    
    def _add_premium_upsell(self):
        """Add premium upsell section for free reports"""
        # Add new page for upsell
        self.pdf.add_page()
        
        # Add section title
        self._set_font('B', 18)
        self.pdf.set_text_color(31, 64, 122)  # Dark blue
        self.pdf.cell(180, 15, "Upgrade to Founding Access", 0, 1, 'C')
        
        # Add upsell description
        self._set_font('', 12)
        self.pdf.set_text_color(0, 0, 0)  # Reset to black
        
        upsell_text = (
            "Unlock the full potential of your financial future with Iso AI Founding Access. "
            "For $149 one-time, you'll receive:"
        )
        
        self.pdf.multi_cell(180, 8, upsell_text, 0, 'C')
        self.pdf.ln(5)
        
        # Add founding access features list
        features = [
            "1 Premium Portfolio Report ($99 value)",
            "Immediate MVP Access with unlimited portfolio assets",
            "6 Months of Premium Free at Launch ($180 value)",
            "Unlimited simulations with up to 5000 simulation paths",
            "30-year time horizon projections (vs. 5 years for free)",
            "Unlimited AI assistant queries (vs. 1 per month)",
            "Founding Supporter Recognition"
        ]
        
        self._set_font('', 11)
        
        for feature in features:
            self.pdf.cell(10, 8, "•", 0, 0)
            self.pdf.cell(170, 8, feature, 0, 1)
        
        self.pdf.ln(10)
        
        # Add call to action
        self._set_font('B', 14)
        self.pdf.cell(180, 10, "Secure Your Founding Access Today", 0, 1, 'C')
        
        self._set_font('', 11)
        cta_text = (
            "Contact hello@isoai.co to upgrade to Founding Access "
            "and join our exclusive community of early supporters."
        )
        
        self.pdf.multi_cell(180, 8, cta_text, 0, 'C')
        self.pdf.ln(5)
        
        # Add QR code
        qr_path = self._create_qr_code("https://isoai.co/founding-access")
        if qr_path:
            self.pdf.image(qr_path, x=75, y=None, w=60)
        
        # Add price and validity
        self.pdf.ln(65)  # Space for QR code
        self._set_font('B', 12)
        self.pdf.cell(180, 10, "Founding Access: $149 (One-Time Payment)", 0, 1, 'C')
        
        self._set_font('', 10)
        self.pdf.cell(180, 8, "Limited time offer - Founding Access spots are limited.", 0, 1, 'C')
    
    def _create_qr_code(self, url: str) -> Optional[str]:
        """Create a QR code for the given URL"""
        try:
            qr = qrcode.QRCode(
                version=1,
                error_correction=qrcode.constants.ERROR_CORRECT_L,
                box_size=10,
                border=4,
            )
            qr.add_data(url)
            qr.make(fit=True)
            
            qr_img = qr.make_image(fill_color="black", back_color="white")
            
            # Save to temporary file
            fd, temp_path = tempfile.mkstemp(suffix='.png')
            qr_img.save(temp_path)
            os.close(fd)
            
            # Add to list of temporary files for cleanup
            self.temp_files.append(temp_path)
            
            return temp_path
        
        except Exception as e:
            logger.error(f"Error creating QR code: {str(e)}")
            return None