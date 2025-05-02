import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import dash_bootstrap_components as dbc
from typing import Dict, Any, List, Optional
import os
import json
import logging

# Import local components
from iso_financial_mvp.dashboard.disclaimer_component import create_disclaimer_component, init_disclaimer_callback

# Import local modules
from iso_financial_mvp.simulation_engines.monte_carlo import MonteCarloSimulator
from iso_financial_mvp.simulation_engines.scenario_analysis import ScenarioAnalysisEngine
from iso_financial_mvp.simulation_engines.risk_parity import RiskParityEngine
from iso_financial_mvp.simulation_engines.cvar import CVaRCalculator
from iso_financial_mvp.data_sources.polygon_client import PolygonDataSource

# Setup logger
logger = logging.getLogger(__name__)

class FinancialDashboard:
    def __init__(self, simulation_results: Optional[Dict[str, Any]] = None, title: str = "ISO Financial Dashboard"):
        """
        Initialize the financial dashboard
        
        Args:
            simulation_results: Optional simulation results to display
            title: Dashboard title
        """
        self.title = title
        self.simulation_results = simulation_results or {}
        self.data_source = PolygonDataSource()
        
        # Initialize Dash app with Bootstrap
        self.app = dash.Dash(__name__, 
                           external_stylesheets=[dbc.themes.BOOTSTRAP],
                           title=title,
                           meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])
        
        # Setup layout
        self.setup_layout()
        
        # Setup callbacks
        self.setup_callbacks()
        
        # Initialize disclaimer callback
        init_disclaimer_callback(self.app)
    
    def setup_layout(self):
        """Configure the dashboard layout"""
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1(self.title, className="dashboard-title"),
                html.P("Interactive financial portfolio analysis", className="dashboard-subtitle")
            ], className="header"),
            
            # Main content container
            html.Div([
                # Left sidebar for controls
                html.Div([
                    html.H3("Portfolio Analysis Controls"),
                    
                    # Time horizon control
                    html.Div([
                        html.Label("Time Horizon (Years)"),
                        dcc.Slider(
                            id="time-horizon-slider",
                            min=1,
                            max=30,
                            step=1,
                            value=10,
                            marks={i: str(i) for i in range(0, 31, 5)},
                        )
                    ], className="control-group"),
                    
                    # Simulation count control
                    html.Div([
                        html.Label("Number of Simulations"),
                        dcc.Dropdown(
                            id="simulation-count-dropdown",
                            options=[
                                {"label": "100 (Faster)", "value": 100},
                                {"label": "500 (Balanced)", "value": 500},
                                {"label": "1000 (More Accurate)", "value": 1000}
                            ],
                            value=500
                        )
                    ], className="control-group"),
                    
                    # Scenario selection
                    html.Div([
                        html.Label("Scenario"),
                        dcc.Dropdown(
                            id="scenario-dropdown",
                            options=[
                                {"label": "Baseline", "value": "baseline"},
                                {"label": "Crypto Winter", "value": "crypto_winter"},
                                {"label": "Regulatory Crackdown", "value": "regulatory_crackdown"},
                                {"label": "Market Recession", "value": "recession"},
                                {"label": "NFT Liquidity Freeze", "value": "nft_liquidity_freeze"},
                                {"label": "ETH Price Crash", "value": "eth_crash"},
                                {"label": "Inflation Surge", "value": "inflation_surge"}
                            ],
                            value="baseline"
                        )
                    ], className="control-group"),
                    
                    # Risk level control
                    html.Div([
                        html.Label("Risk Profile"),
                        dcc.RadioItems(
                            id="risk-profile-radio",
                            options=[
                                {"label": "Conservative", "value": "conservative"},
                                {"label": "Moderate", "value": "moderate"},
                                {"label": "Aggressive", "value": "aggressive"}
                            ],
                            value="moderate"
                        )
                    ], className="control-group"),
                    
                    # Run simulation button
                    html.Div([
                        html.Button("Run Simulation", id="run-simulation-button", className="button")
                    ], className="control-group"),
                    
                    # Portfolio composition table
                    html.Div([
                        html.H4("Portfolio Composition"),
                        html.Div(id="portfolio-table-container")
                    ], className="control-group"),
                    
                ], className="sidebar"),
                
                # Main content area
                html.Div([
                    # Tabs for different visualizations
                    dcc.Tabs([
                        # Monte Carlo simulation tab
                        dcc.Tab(label="Monte Carlo Simulation", children=[
                            html.Div([
                                dcc.Graph(id="monte-carlo-chart"),
                                html.Div([
                                    html.Div([
                                        html.H4("Simulation Statistics"),
                                        html.Div(id="monte-carlo-stats")
                                    ], className="stats-box")
                                ], className="stats-container")
                            ], className="tab-content")
                        ]),
                        
                        # Risk analysis tab
                        dcc.Tab(label="Risk Analysis", children=[
                            html.Div([
                                html.Div([
                                    dcc.Graph(id="risk-contribution-chart")
                                ], className="chart-container"),
                                
                                html.Div([
                                    html.Div([
                                        html.H4("Value at Risk (VaR)"),
                                        html.Div(id="var-stats")
                                    ], className="stats-box"),
                                    
                                    html.Div([
                                        html.H4("Maximum Drawdown"),
                                        html.Div(id="drawdown-stats")
                                    ], className="stats-box")
                                ], className="stats-container"),
                                
                                dcc.Graph(id="risk-distribution-chart")
                            ], className="tab-content")
                        ]),
                        
                        # Scenario analysis tab
                        dcc.Tab(label="Scenario Analysis", children=[
                            html.Div([
                                dcc.Graph(id="scenario-comparison-chart"),
                                
                                html.Div([
                                    html.Div([
                                        html.H4("Scenario Impact"),
                                        html.Div(id="scenario-impact-stats")
                                    ], className="stats-box"),
                                    
                                    html.Div([
                                        html.H4("Recovery Analysis"),
                                        html.Div(id="recovery-stats")
                                    ], className="stats-box")
                                ], className="stats-container"),
                                
                                dcc.Graph(id="scenario-probability-chart")
                            ], className="tab-content")
                        ]),
                        
                        # Asset allocation tab
                        dcc.Tab(label="Asset Allocation", children=[
                            html.Div([
                                html.Div([
                                    dcc.Graph(id="current-allocation-chart"),
                                    dcc.Graph(id="optimal-allocation-chart")
                                ], className="chart-container"),
                                
                                html.Div([
                                    html.H4("Recommended Trades"),
                                    html.Div(id="recommended-trades")
                                ], className="stats-box")
                            ], className="tab-content")
                        ]),
                        
                        # Historical data tab
                        dcc.Tab(label="Historical Data", children=[
                            html.Div([
                                html.Div([
                                    html.Label("Select Assets to Compare"),
                                    dcc.Dropdown(
                                        id="historical-assets-dropdown",
                                        multi=True
                                    )
                                ], className="control-group"),
                                
                                html.Div([
                                    html.Label("Time Period"),
                                    dcc.Dropdown(
                                        id="historical-period-dropdown",
                                        options=[
                                            {"label": "1 Month", "value": "1mo"},
                                            {"label": "3 Months", "value": "3mo"},
                                            {"label": "6 Months", "value": "6mo"},
                                            {"label": "1 Year", "value": "1y"},
                                            {"label": "3 Years", "value": "3y"},
                                            {"label": "5 Years", "value": "5y"},
                                            {"label": "10 Years", "value": "10y"}
                                        ],
                                        value="1y"
                                    )
                                ], className="control-group"),
                                
                                dcc.Graph(id="historical-price-chart"),
                                dcc.Graph(id="historical-return-chart")
                            ], className="tab-content")
                        ])
                    ], id="dashboard-tabs")
                ], className="main-content")
            ], className="content-container"),
            
            # Hidden div for storing intermediate data
            html.Div(id="simulation-results-store", style={"display": "none"}),
            html.Div(id="portfolio-data-store", style={"display": "none"}, children="{}"),
            
            # Footer with disclaimer
            html.Div([
                html.P("ISO Financial Dashboard | Interactive Portfolio Analysis"),
                create_disclaimer_component(id="disclaimer-collapse", is_open=False)
            ], className="footer")
        ], className="dashboard-container")
    
    def setup_callbacks(self):
        """Set up interactive callbacks"""
        
        @self.app.callback(
            [Output("simulation-results-store", "children"),
             Output("monte-carlo-chart", "figure"),
             Output("monte-carlo-stats", "children"),
             Output("risk-contribution-chart", "figure"),
             Output("var-stats", "children"),
             Output("drawdown-stats", "children"),
             Output("risk-distribution-chart", "figure"),
             Output("scenario-comparison-chart", "figure"),
             Output("scenario-impact-stats", "children"),
             Output("recovery-stats", "children"),
             Output("scenario-probability-chart", "figure"),
             Output("current-allocation-chart", "figure"),
             Output("optimal-allocation-chart", "figure"),
             Output("recommended-trades", "children")],
            [Input("run-simulation-button", "n_clicks")],
            [State("time-horizon-slider", "value"),
             State("simulation-count-dropdown", "value"),
             State("scenario-dropdown", "value"),
             State("risk-profile-radio", "value"),
             State("portfolio-data-store", "children")]
        )
        def run_simulation_pipeline(n_clicks, time_horizon, sim_count, scenario_key, risk_profile, portfolio_data_json):
            """Run the full simulation pipeline and update all charts"""
            if n_clicks is None:
                # Initialize with empty or placeholder figures
                return self._get_empty_outputs()
            
            # Parse portfolio data
            try:
                portfolio_data = json.loads(portfolio_data_json)
            except:
                portfolio_data = self._get_default_portfolio()
            
            # If portfolio is empty, use default
            if not portfolio_data.get("assets"):
                portfolio_data = self._get_default_portfolio()
            
            # Setup timeframe
            timeframe = {"unit": "years", "value": time_horizon}
            
            # Initialize simulators
            monte_carlo = MonteCarloSimulator(num_simulations=sim_count)
            scenario_engine = ScenarioAnalysisEngine(base_simulator=monte_carlo)
            risk_parity = RiskParityEngine()
            cvar_calculator = CVaRCalculator(confidence_level=0.95)
            
            # Run Monte Carlo simulation
            mc_results = monte_carlo.simulate_portfolio(
                portfolio_config=portfolio_data,
                timeframe=timeframe,
                risk_profile=risk_profile
            )
            
            # Run scenario analysis if not baseline
            if scenario_key != "baseline":
                scenario = scenario_engine.get_prebuilt_scenario(scenario_key)
                if scenario:
                    scenario_results = scenario_engine.run_scenario_analysis(
                        base_portfolio_config=portfolio_data,
                        timeframe=timeframe,
                        risk_profile=risk_profile,
                        scenario=scenario
                    )
                else:
                    # Fallback to a default scenario
                    scenario = {"name": "Default Scenario", "description": "Fallback scenario", "adjustments": {}}
                    scenario_results = scenario_engine.run_scenario_analysis(
                        base_portfolio_config=portfolio_data,
                        timeframe=timeframe,
                        risk_profile=risk_profile,
                        scenario=scenario
                    )
            else:
                # Create empty scenario results structure for baseline
                scenario_results = {
                    "baseline": mc_results,
                    "scenario": mc_results,
                    "comparison": {
                        "mean_change": {"percentage": 0, "absolute": 0},
                        "median_change": {"percentage": 0, "absolute": 0},
                        "risk_measures": {
                            "var_change": {"percentage": 0, "absolute": 0},
                            "cvar_change": {"percentage": 0, "absolute": 0},
                            "drawdown_change": {"percentage": 0, "absolute": 0}
                        },
                        "probability_of_loss": {
                            "probability_any_loss": {"baseline": 0.1, "scenario": 0.1, "difference": 0},
                            "probability_10pct_loss": {"baseline": 0.05, "scenario": 0.05, "difference": 0},
                            "probability_25pct_loss": {"baseline": 0.02, "scenario": 0.02, "difference": 0}
                        },
                        "recovery_analysis": {
                            "paths_that_recover_pct": 0.95,
                            "avg_recovery_time_months": 6,
                            "median_recovery_time_months": 5
                        }
                    },
                    "scenario_details": {
                        "name": "Baseline",
                        "description": "No adjustments to baseline assumptions"
                    }
                }
            
            # Calculate VaR and CVaR
            cvar_results = cvar_calculator.calculate_var_cvar(
                simulation_paths=mc_results["simulation_paths"],
                initial_investment=portfolio_data.get("initial_investment", 100000)
            )
            
            # Calculate risk parity allocation
            assets_with_vol = []
            for asset in portfolio_data.get("assets", []):
                asset_type = asset.get("type", "")
                asset_name = asset.get("name", "")
                
                # Extract volatility from Monte Carlo parameters if available
                volatility = 0.2  # Default
                
                assets_with_vol.append({
                    "name": asset_name,
                    "type": asset_type,
                    "volatility": volatility
                })
            
            # Get current weights
            current_weights = [asset.get("weight", 1.0 / len(portfolio_data.get("assets", []))) 
                              for asset in portfolio_data.get("assets", [])]
            
            # Calculate risk parity allocation
            rp_results = risk_parity.calculate_risk_parity_allocation(
                portfolio_assets=assets_with_vol
            )
            
            # Generate trade recommendations
            trades = risk_parity.recommend_trades(
                current_weights=current_weights,
                target_weights=rp_results["weights"],
                asset_details=rp_results["asset_details"],
                total_portfolio_value=portfolio_data.get("initial_investment", 100000)
            )
            
            # Store all results
            all_results = {
                "monte_carlo": mc_results,
                "scenario": scenario_results,
                "cvar": cvar_results,
                "risk_parity": {**rp_results, "trades": trades}
            }
            
            # Create all figures and output components
            outputs = self._create_dashboard_outputs(all_results, portfolio_data)
            
            # Add the JSON results as first output
            outputs.insert(0, json.dumps(all_results))
            
            return outputs
        
        @self.app.callback(
            Output("portfolio-table-container", "children"),
            [Input("portfolio-data-store", "children")]
        )
        def update_portfolio_table(portfolio_data_json):
            """Update the portfolio composition table"""
            try:
                portfolio_data = json.loads(portfolio_data_json)
            except:
                portfolio_data = self._get_default_portfolio()
            
            # If portfolio is empty, use default
            if not portfolio_data.get("assets"):
                portfolio_data = self._get_default_portfolio()
            
            # Create portfolio table
            assets = portfolio_data.get("assets", [])
            
            if not assets:
                return html.P("No assets in portfolio")
            
            table_rows = [html.Tr([
                html.Th("Asset Type"),
                html.Th("Asset Name"),
                html.Th("Weight")
            ])]
            
            for asset in assets:
                table_rows.append(html.Tr([
                    html.Td(asset.get("type", "")),
                    html.Td(asset.get("name", "")),
                    html.Td(f"{asset.get('weight', 0) * 100:.1f}%")
                ]))
            
            return html.Table(table_rows, className="portfolio-table")
        
        @self.app.callback(
            [Output("historical-assets-dropdown", "options"),
             Output("historical-assets-dropdown", "value")],
            [Input("portfolio-data-store", "children")]
        )
        def update_historical_assets_dropdown(portfolio_data_json):
            """Update the assets dropdown for historical comparison"""
            try:
                portfolio_data = json.loads(portfolio_data_json)
            except:
                portfolio_data = self._get_default_portfolio()
            
            # If portfolio is empty, use default
            if not portfolio_data.get("assets"):
                portfolio_data = self._get_default_portfolio()
            
            assets = portfolio_data.get("assets", [])
            
            options = []
            for asset in assets:
                asset_type = asset.get("type", "")
                asset_name = asset.get("name", "")
                
                # Get ticker
                ticker = self.data_source.get_ticker_for_asset(asset_type, asset_name)
                if ticker:
                    options.append({
                        "label": f"{asset_name} ({asset_type.capitalize()})",
                        "value": f"{asset_type}|{asset_name}"
                    })
            
            # Default to first two assets or fewer if not enough assets
            default_value = [option["value"] for option in options[:min(2, len(options))]]
            
            return options, default_value
        
        @self.app.callback(
            [Output("historical-price-chart", "figure"),
             Output("historical-return-chart", "figure")],
            [Input("historical-assets-dropdown", "value"),
             Input("historical-period-dropdown", "value")]
        )
        def update_historical_charts(selected_assets, period):
            """Update historical price and return charts"""
            if not selected_assets:
                # Return empty figures
                empty_fig = go.Figure()
                empty_fig.update_layout(
                    title="No assets selected",
                    xaxis_title="Date",
                    yaxis_title="Value"
                )
                return empty_fig, empty_fig
            
            # Get data for each asset
            all_prices = {}
            all_returns = {}
            
            for asset_id in selected_assets:
                # Parse asset_id
                try:
                    asset_type, asset_name = asset_id.split("|")
                    
                    # Get historical data
                    data = self.data_source.get_historical_data(
                        asset_type=asset_type,
                        asset_name=asset_name,
                        period=period
                    )
                    
                    if data is not None and not data.empty:
                        # Store prices
                        all_prices[f"{asset_name} ({asset_type})"] = data["Adj Close"]
                        
                        # Calculate returns
                        returns = data["Adj Close"].pct_change().dropna()
                        all_returns[f"{asset_name} ({asset_type})"] = returns
                except Exception as e:
                    logger.error(f"Error processing historical data for {asset_id}: {str(e)}")
            
            # Create price chart
            if all_prices:
                price_df = pd.DataFrame(all_prices)
                
                # Normalize to starting value of 100
                for col in price_df.columns:
                    price_df[col] = price_df[col] / price_df[col].iloc[0] * 100
                
                price_fig = px.line(
                    price_df, 
                    title="Historical Price Comparison (Normalized)",
                    labels={"value": "Value (Normalized to 100)", "variable": "Asset"}
                )
                
                price_fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Value (Starting = 100)",
                    legend_title="Asset"
                )
            else:
                price_fig = go.Figure()
                price_fig.update_layout(
                    title="No historical price data available",
                    xaxis_title="Date",
                    yaxis_title="Value"
                )
            
            # Create returns chart
            if all_returns:
                return_df = pd.DataFrame(all_returns)
                
                return_fig = px.histogram(
                    return_df, 
                    barmode="overlay",
                    title="Return Distribution",
                    labels={"value": "Daily Return", "variable": "Asset"},
                    opacity=0.7
                )
                
                return_fig.update_layout(
                    xaxis_title="Daily Return",
                    yaxis_title="Frequency",
                    legend_title="Asset"
                )
            else:
                return_fig = go.Figure()
                return_fig.update_layout(
                    title="No historical return data available",
                    xaxis_title="Return",
                    yaxis_title="Frequency"
                )
            
            return price_fig, return_fig
    
    def _get_default_portfolio(self) -> Dict[str, Any]:
        """Get a default portfolio configuration"""
        return {
            "assets": [
                {"type": "crypto", "name": "bitcoin", "weight": 0.2},
                {"type": "crypto", "name": "ethereum", "weight": 0.2},
                {"type": "traditional", "name": "us_stocks", "weight": 0.3},
                {"type": "traditional", "name": "bonds", "weight": 0.2},
                {"type": "alternative", "name": "gold", "weight": 0.1}
            ],
            "initial_investment": 100000
        }
    
    def _get_empty_outputs(self):
        """Get empty/placeholder outputs for initial dashboard state"""
        # Create empty figures
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Run simulation to see results",
            xaxis_title="Time",
            yaxis_title="Value"
        )
        
        # Create empty stats
        empty_stats = html.Div([
            html.P("Run simulation to see statistics")
        ])
        
        # Create empty recommended trades
        empty_trades = html.Div([
            html.P("Run simulation to see trade recommendations")
        ])
        
        # Bundle all outputs
        return [
            "{}",  # Empty JSON results
            empty_fig,  # monte_carlo_chart
            empty_stats,  # monte_carlo_stats
            empty_fig,  # risk_contribution_chart
            empty_stats,  # var_stats
            empty_stats,  # drawdown_stats
            empty_fig,  # risk_distribution_chart
            empty_fig,  # scenario_comparison_chart
            empty_stats,  # scenario_impact_stats
            empty_stats,  # recovery_stats
            empty_fig,  # scenario_probability_chart
            empty_fig,  # current_allocation_chart
            empty_fig,  # optimal_allocation_chart
            empty_trades  # recommended_trades
        ]
    
    def _create_dashboard_outputs(self, results: Dict[str, Any], portfolio_data: Dict[str, Any]) -> List[Any]:
        """Create all dashboard outputs from simulation results"""
        outputs = []
        
        # 1. Monte Carlo Chart
        mc_fig = self._create_monte_carlo_chart(results.get("monte_carlo", {}))
        outputs.append(mc_fig)
        
        # 2. Monte Carlo Stats
        mc_stats = self._create_monte_carlo_stats(results.get("monte_carlo", {}))
        outputs.append(mc_stats)
        
        # 3. Risk Contribution Chart
        risk_contrib_fig = self._create_risk_contribution_chart(results.get("risk_parity", {}))
        outputs.append(risk_contrib_fig)
        
        # 4. VaR Stats
        var_stats = self._create_var_stats(results.get("cvar", {}))
        outputs.append(var_stats)
        
        # 5. Drawdown Stats
        drawdown_stats = self._create_drawdown_stats(results.get("monte_carlo", {}))
        outputs.append(drawdown_stats)
        
        # 6. Risk Distribution Chart
        risk_dist_fig = self._create_risk_distribution_chart(results.get("monte_carlo", {}))
        outputs.append(risk_dist_fig)
        
        # 7. Scenario Comparison Chart
        scenario_comp_fig = self._create_scenario_comparison_chart(results.get("scenario", {}))
        outputs.append(scenario_comp_fig)
        
        # 8. Scenario Impact Stats
        scenario_impact_stats = self._create_scenario_impact_stats(results.get("scenario", {}))
        outputs.append(scenario_impact_stats)
        
        # 9. Recovery Stats
        recovery_stats = self._create_recovery_stats(results.get("scenario", {}))
        outputs.append(recovery_stats)
        
        # 10. Scenario Probability Chart
        scenario_prob_fig = self._create_scenario_probability_chart(results.get("scenario", {}))
        outputs.append(scenario_prob_fig)
        
        # 11. Current Allocation Chart
        current_alloc_fig = self._create_current_allocation_chart(portfolio_data)
        outputs.append(current_alloc_fig)
        
        # 12. Optimal Allocation Chart
        optimal_alloc_fig = self._create_optimal_allocation_chart(results.get("risk_parity", {}))
        outputs.append(optimal_alloc_fig)
        
        # 13. Recommended Trades
        recommended_trades = self._create_recommended_trades(results.get("risk_parity", {}).get("trades", {}))
        outputs.append(recommended_trades)
        
        return outputs
    
    def _create_monte_carlo_chart(self, mc_results: Dict[str, Any]) -> go.Figure:
        """Create Monte Carlo simulation paths chart"""
        paths = mc_results.get("simulation_paths")
        
        if paths is None:
            # Return empty figure
            fig = go.Figure()
            fig.update_layout(
                title="No Monte Carlo simulation data available",
                xaxis_title="Time",
                yaxis_title="Portfolio Value"
            )
            return fig
        
        # Convert to numpy array if it's a list
        if isinstance(paths, list):
            paths = np.array(paths)
        
        # Get number of time steps
        _, num_steps = paths.shape
        
        # Create figure
        fig = go.Figure()
        
        # Plot a subset of paths (for better visualization)
        num_paths_to_plot = min(100, len(paths))
        path_indices = np.random.choice(len(paths), num_paths_to_plot, replace=False)
        
        for i in path_indices:
            fig.add_trace(go.Scatter(
                y=paths[i],
                mode='lines',
                line=dict(width=0.5, color='rgba(70, 130, 180, 0.1)'),
                showlegend=False
            ))
        
        # Add percentile lines
        percentiles = [5, 50, 95]
        colors = ['rgba(255, 0, 0, 0.7)', 'rgba(0, 0, 0, 0.9)', 'rgba(0, 128, 0, 0.7)']
        names = ['5th Percentile', 'Median', '95th Percentile']
        
        for p, color, name in zip(percentiles, colors, names):
            percentile_values = np.percentile(paths, p, axis=0)
            
            fig.add_trace(go.Scatter(
                y=percentile_values,
                mode='lines',
                line=dict(width=2, color=color),
                name=name
            ))
        
        # Layout
        fig.update_layout(
            title='Monte Carlo Simulation of Portfolio Value',
            xaxis_title='Time (Months)',
            yaxis_title='Portfolio Value ($)',
            hovermode='closest',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        # Create x-axis values (months)
        month_labels = list(range(0, num_steps))
        fig.update_xaxes(tickvals=month_labels[::12], ticktext=[str(m//12) + "y" for m in month_labels[::12]])
        
        return fig
    
    def _create_monte_carlo_stats(self, mc_results: Dict[str, Any]) -> html.Div:
        """Create Monte Carlo statistics display"""
        if not mc_results or "statistics" not in mc_results:
            return html.Div([
                html.P("No Monte Carlo statistics available")
            ])
        
        stats = mc_results.get("statistics", {})
        confidence_intervals = stats.get("confidence_intervals", {})
        
        # Format the statistics
        stats_div = html.Div([
            html.Table([
                html.Tr([
                    html.Th("Metric"),
                    html.Th("Value")
                ]),
                html.Tr([
                    html.Td("Mean Final Value"),
                    html.Td(f"${stats.get('mean', 0):,.2f}")
                ]),
                html.Tr([
                    html.Td("Median Final Value"),
                    html.Td(f"${stats.get('median', 0):,.2f}")
                ]),
                html.Tr([
                    html.Td("Standard Deviation"),
                    html.Td(f"${stats.get('std', 0):,.2f}")
                ]),
                html.Tr([
                    html.Td("5th Percentile"),
                    html.Td(f"${confidence_intervals.get('p5', 0):,.2f}")
                ]),
                html.Tr([
                    html.Td("95th Percentile"),
                    html.Td(f"${confidence_intervals.get('p95', 0):,.2f}")
                ])
            ], className="stats-table")
        ])
        
        return stats_div
    
    def _create_risk_contribution_chart(self, rp_results: Dict[str, Any]) -> go.Figure:
        """Create risk contribution pie chart"""
        if not rp_results or "asset_details" not in rp_results:
            # Return empty figure
            fig = go.Figure()
            fig.update_layout(
                title="No risk contribution data available",
                xaxis_title="Asset",
                yaxis_title="Risk Contribution"
            )
            return fig
        
        asset_details = rp_results.get("asset_details", [])
        
        # Extract data for chart
        labels = [f"{detail.get('name')} ({detail.get('type')})" for detail in asset_details]
        values = [detail.get("risk_contribution_pct", 0) for detail in asset_details]
        
        # Create pie chart
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=.3,
            textinfo='label+percent'
        )])
        
        fig.update_layout(
            title='Risk Contribution by Asset',
            legend=dict(
                yanchor="bottom",
                y=0.01,
                xanchor="right",
                x=0.99
            )
        )
        
        return fig
    
    def _create_var_stats(self, cvar_results: Dict[str, Any]) -> html.Div:
        """Create Value at Risk statistics display"""
        if not cvar_results:
            return html.Div([
                html.P("No Value at Risk statistics available")
            ])
        
        # Format the statistics
        stats_div = html.Div([
            html.Table([
                html.Tr([
                    html.Th("Metric"),
                    html.Th("Value")
                ]),
                html.Tr([
                    html.Td("Value at Risk (95%)"),
                    html.Td(f"${cvar_results.get('var_amount', 0):,.2f}")
                ]),
                html.Tr([
                    html.Td("Conditional VaR (95%)"),
                    html.Td(f"${cvar_results.get('cvar_amount', 0):,.2f}")
                ]),
                html.Tr([
                    html.Td("Worst Loss"),
                    html.Td(f"${cvar_results.get('worst_loss', 0):,.2f}")
                ]),
                html.Tr([
                    html.Td("Probability of Loss"),
                    html.Td(f"{cvar_results.get('probability_of_loss', 0) * 100:.1f}%")
                ])
            ], className="stats-table")
        ])
        
        return stats_div
    
    def _create_drawdown_stats(self, mc_results: Dict[str, Any]) -> html.Div:
        """Create drawdown statistics display"""
        if not mc_results or "statistics" not in mc_results:
            return html.Div([
                html.P("No drawdown statistics available")
            ])
        
        stats = mc_results.get("statistics", {})
        
        # Format the statistics
        stats_div = html.Div([
            html.Table([
                html.Tr([
                    html.Th("Metric"),
                    html.Th("Value")
                ]),
                html.Tr([
                    html.Td("Average Maximum Drawdown"),
                    html.Td(f"{stats.get('avg_max_drawdown', 0) * 100:.1f}%")
                ]),
                html.Tr([
                    html.Td("Worst Drawdown"),
                    html.Td(f"{stats.get('max_drawdown', 0) * 100:.1f}%")
                ])
            ], className="stats-table")
        ])
        
        return stats_div
    
    def _create_risk_distribution_chart(self, mc_results: Dict[str, Any]) -> go.Figure:
        """Create final value distribution histogram"""
        paths = mc_results.get("simulation_paths")
        
        if paths is None:
            # Return empty figure
            fig = go.Figure()
            fig.update_layout(
                title="No distribution data available",
                xaxis_title="Final Value",
                yaxis_title="Frequency"
            )
            return fig
        
        # Convert to numpy array if it's a list
        if isinstance(paths, list):
            paths = np.array(paths)
        
        # Get final values
        final_values = paths[:, -1]
        
        # Create histogram
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=final_values,
            histnorm='probability',
            marker_color='rgba(0, 0, 128, 0.7)',
            name='Final Value Distribution'
        ))
        
        # Add statistics lines
        stats = mc_results.get("statistics", {})
        
        if stats:
            mean_val = stats.get("mean", 0)
            median_val = stats.get("median", 0)
            var_95 = stats.get("var_95", 0)
            
            # Add mean line
            fig.add_vline(
                x=mean_val,
                line_dash="solid",
                line_color="green",
                annotation_text="Mean",
                annotation_position="top right"
            )
            
            # Add median line
            fig.add_vline(
                x=median_val,
                line_dash="dash",
                line_color="blue",
                annotation_text="Median",
                annotation_position="bottom right"
            )
            
            # Add VaR line
            if var_95 > 0:
                fig.add_vline(
                    x=var_95,
                    line_dash="dot",
                    line_color="red",
                    annotation_text="VaR (95%)",
                    annotation_position="top left"
                )
        
        # Layout
        fig.update_layout(
            title='Final Portfolio Value Distribution',
            xaxis_title='Portfolio Value ($)',
            yaxis_title='Probability',
            bargap=0.1
        )
        
        return fig
    
    def _create_scenario_comparison_chart(self, scenario_results: Dict[str, Any]) -> go.Figure:
        """Create scenario comparison chart"""
        if not scenario_results or "comparison" not in scenario_results:
            # Return empty figure
            fig = go.Figure()
            fig.update_layout(
                title="No scenario comparison data available",
                xaxis_title="Metric",
                yaxis_title="Value"
            )
            return fig
        
        comparison = scenario_results.get("comparison", {})
        scenario_name = scenario_results.get("scenario_details", {}).get("name", "Scenario")
        
        # Extract key metrics for comparison
        metrics = [
            "Mean Return",
            "Median Return",
            "Value at Risk",
            "Conditional VaR",
            "Max Drawdown"
        ]
        
        # Extract values
        baseline_values = [
            0,  # Baseline mean (reference point)
            0,  # Baseline median (reference point)
            0,  # Baseline VaR (reference point)
            0,  # Baseline CVaR (reference point)
            0   # Baseline drawdown (reference point)
        ]
        
        scenario_values = [
            comparison.get("mean_change", {}).get("percentage", 0),
            comparison.get("median_change", {}).get("percentage", 0),
            comparison.get("risk_measures", {}).get("var_change", {}).get("percentage", 0),
            comparison.get("risk_measures", {}).get("cvar_change", {}).get("percentage", 0),
            comparison.get("risk_measures", {}).get("drawdown_change", {}).get("percentage", 0)
        ]
        
        # Create figure
        fig = go.Figure()
        
        # Add baseline bar
        fig.add_trace(go.Bar(
            x=metrics,
            y=baseline_values,
            name='Baseline',
            marker_color='rgba(0, 0, 255, 0.7)'
        ))
        
        # Add scenario bar
        fig.add_trace(go.Bar(
            x=metrics,
            y=scenario_values,
            name=scenario_name,
            marker_color='rgba(255, 0, 0, 0.7)'
        ))
        
        # Layout
        fig.update_layout(
            title=f'Impact of {scenario_name} on Portfolio Metrics',
            xaxis_title='Metric',
            yaxis_title='Change (%)',
            barmode='group',
            yaxis_tickformat='.1%',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            )
        )
        
        return fig
    
    def _create_scenario_impact_stats(self, scenario_results: Dict[str, Any]) -> html.Div:
        """Create scenario impact statistics display"""
        if not scenario_results or "comparison" not in scenario_results:
            return html.Div([
                html.P("No scenario impact statistics available")
            ])
        
        comparison = scenario_results.get("comparison", {})
        scenario_name = scenario_results.get("scenario_details", {}).get("name", "Scenario")
        
        # Format the statistics
        stats_div = html.Div([
            html.Table([
                html.Tr([
                    html.Th("Metric"),
                    html.Th("Change")
                ]),
                html.Tr([
                    html.Td("Mean Return"),
                    html.Td(f"{comparison.get('mean_change', {}).get('percentage', 0) * 100:.1f}%")
                ]),
                html.Tr([
                    html.Td("Median Return"),
                    html.Td(f"{comparison.get('median_change', {}).get('percentage', 0) * 100:.1f}%")
                ]),
                html.Tr([
                    html.Td("Value at Risk"),
                    html.Td(f"{comparison.get('risk_measures', {}).get('var_change', {}).get('percentage', 0) * 100:.1f}%")
                ]),
                html.Tr([
                    html.Td("Conditional VaR"),
                    html.Td(f"{comparison.get('risk_measures', {}).get('cvar_change', {}).get('percentage', 0) * 100:.1f}%")
                ])
            ], className="stats-table")
        ])
        
        return stats_div
    
    def _create_recovery_stats(self, scenario_results: Dict[str, Any]) -> html.Div:
        """Create recovery statistics display"""
        if not scenario_results or "comparison" not in scenario_results:
            return html.Div([
                html.P("No recovery statistics available")
            ])
        
        recovery = scenario_results.get("comparison", {}).get("recovery_analysis", {})
        
        # Format the statistics
        stats_div = html.Div([
            html.Table([
                html.Tr([
                    html.Th("Metric"),
                    html.Th("Value")
                ]),
                html.Tr([
                    html.Td("Paths that Recover"),
                    html.Td(f"{recovery.get('paths_that_recover_pct', 0) * 100:.1f}%")
                ]),
                html.Tr([
                    html.Td("Average Recovery Time"),
                    html.Td(f"{recovery.get('avg_recovery_time_months', 0):.1f} months")
                ]),
                html.Tr([
                    html.Td("Median Recovery Time"),
                    html.Td(f"{recovery.get('median_recovery_time_months', 0):.1f} months")
                ])
            ], className="stats-table")
        ])
        
        return stats_div
    
    def _create_scenario_probability_chart(self, scenario_results: Dict[str, Any]) -> go.Figure:
        """Create scenario probability chart"""
        if not scenario_results or "comparison" not in scenario_results:
            # Return empty figure
            fig = go.Figure()
            fig.update_layout(
                title="No scenario probability data available",
                xaxis_title="Loss Threshold",
                yaxis_title="Probability"
            )
            return fig
        
        probability_data = scenario_results.get("comparison", {}).get("probability_of_loss", {})
        scenario_name = scenario_results.get("scenario_details", {}).get("name", "Scenario")
        
        if not probability_data:
            # Return empty figure
            fig = go.Figure()
            fig.update_layout(
                title="No scenario probability data available",
                xaxis_title="Loss Threshold",
                yaxis_title="Probability"
            )
            return fig
        
        # Extract data
        thresholds = ["Any Loss", "10% Loss", "25% Loss", "50% Loss"]
        baseline_probs = [
            probability_data.get("probability_any_loss", {}).get("baseline", 0),
            probability_data.get("probability_10pct_loss", {}).get("baseline", 0),
            probability_data.get("probability_25pct_loss", {}).get("baseline", 0),
            probability_data.get("probability_50pct_loss", {}).get("baseline", 0)
        ]
        
        scenario_probs = [
            probability_data.get("probability_any_loss", {}).get("scenario", 0),
            probability_data.get("probability_10pct_loss", {}).get("scenario", 0),
            probability_data.get("probability_25pct_loss", {}).get("scenario", 0),
            probability_data.get("probability_50pct_loss", {}).get("scenario", 0)
        ]
        
        # Create figure
        fig = go.Figure()
        
        # Add baseline bar
        fig.add_trace(go.Bar(
            x=thresholds,
            y=baseline_probs,
            name='Baseline',
            marker_color='rgba(0, 0, 255, 0.7)'
        ))
        
        # Add scenario bar
        fig.add_trace(go.Bar(
            x=thresholds,
            y=scenario_probs,
            name=scenario_name,
            marker_color='rgba(255, 0, 0, 0.7)'
        ))
        
        # Layout
        fig.update_layout(
            title='Probability of Different Loss Thresholds',
            xaxis_title='Loss Threshold',
            yaxis_title='Probability',
            barmode='group',
            yaxis_tickformat='.1%',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            )
        )
        
        return fig
    
    def _create_current_allocation_chart(self, portfolio_data: Dict[str, Any]) -> go.Figure:
        """Create current portfolio allocation pie chart"""
        assets = portfolio_data.get("assets", [])
        
        if not assets:
            # Return empty figure
            fig = go.Figure()
            fig.update_layout(
                title="No portfolio allocation data available",
                xaxis_title="Asset",
                yaxis_title="Weight"
            )
            return fig
        
        # Extract data for chart
        labels = [f"{asset.get('name')} ({asset.get('type')})" for asset in assets]
        values = [asset.get("weight", 1.0 / len(assets)) for asset in assets]
        
        # Create pie chart
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=.3,
            textinfo='label+percent'
        )])
        
        fig.update_layout(
            title='Current Portfolio Allocation',
            legend=dict(
                yanchor="bottom",
                y=0.01,
                xanchor="right",
                x=0.99
            )
        )
        
        return fig
    
    def _create_optimal_allocation_chart(self, rp_results: Dict[str, Any]) -> go.Figure:
        """Create optimal (risk parity) allocation pie chart"""
        if not rp_results or "asset_details" not in rp_results:
            # Return empty figure
            fig = go.Figure()
            fig.update_layout(
                title="No optimal allocation data available",
                xaxis_title="Asset",
                yaxis_title="Weight"
            )
            return fig
        
        asset_details = rp_results.get("asset_details", [])
        
        # Extract data for chart
        labels = [f"{detail.get('name')} ({detail.get('type')})" for detail in asset_details]
        values = [detail.get("weight", 0) for detail in asset_details]
        
        # Create pie chart
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=.3,
            textinfo='label+percent'
        )])
        
        fig.update_layout(
            title='Risk-Parity Optimized Allocation',
            legend=dict(
                yanchor="bottom",
                y=0.01,
                xanchor="right",
                x=0.99
            )
        )
        
        return fig
    
    def _create_recommended_trades(self, trades_data: Dict[str, Any]) -> html.Div:
        """Create recommended trades display"""
        trades = trades_data.get("trades", [])
        
        if not trades:
            return html.Div([
                html.P("No trade recommendations available")
            ])
        
        # Format the trades
        trades_rows = [html.Tr([
            html.Th("Asset"),
            html.Th("Type"),
            html.Th("Action"),
            html.Th("Amount")
        ])]
        
        for trade in trades:
            trades_rows.append(html.Tr([
                html.Td(trade.get("asset_name", "")),
                html.Td(trade.get("asset_type", "")),
                html.Td(trade.get("direction", "").capitalize()),
                html.Td(f"${trade.get('amount', 0):,.2f}")
            ]))
        
        trades_div = html.Div([
            html.Table(trades_rows, className="trades-table"),
            html.P([
                "Estimated Risk Reduction: ",
                html.Span(f"{trades_data.get('estimated_risk_reduction', 0) * 100:.1f}%")
            ], className="risk-reduction")
        ])
        
        return trades_div
    
    def update_portfolio(self, portfolio_data: Dict[str, Any]):
        """Update the portfolio data in the dashboard"""
        if not portfolio_data or not portfolio_data.get("assets"):
            portfolio_data = self._get_default_portfolio()
        
        # Store in the hidden div
        portfolio_json = json.dumps(portfolio_data)
        
        # In a real application, you would need to use a callback or session-based storage
        # For now, we'll just return the JSON to be manually added to the div
        return portfolio_json
    
    def update_simulation_results(self, results: Dict[str, Any]):
        """Update the simulation results in the dashboard"""
        if not results:
            results = {}
        
        # Store in the hidden div
        results_json = json.dumps(results)
        
        # In a real application, you would need to use a callback or session-based storage
        # For now, we'll just return the JSON to be manually added to the div
        return results_json
    
    def run_server(self, debug=True, port=8050):
        """Run the Dash server"""
        self.app.run_server(debug=debug, port=port)

# Create standalone dashboard app
def create_dashboard_app():
    """Create and configure the dashboard app"""
    dashboard = FinancialDashboard(title="ISO Financial Dashboard")
    return dashboard.app

# Run standalone dashboard
if __name__ == "__main__":
    dashboard = FinancialDashboard()
    dashboard.run_server()