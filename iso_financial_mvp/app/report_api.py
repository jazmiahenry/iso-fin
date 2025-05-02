import os
import tempfile
import logging
from typing import Dict, Any, List, Optional
from flask import Blueprint, request, jsonify, send_file

from iso_financial_mvp.llm_interface.query_parser import QueryParser
from iso_financial_mvp.simulation_engines.monte_carlo import MonteCarloSimulator
from iso_financial_mvp.simulation_engines.scenario_analysis import ScenarioAnalysisEngine
from iso_financial_mvp.simulation_engines.risk_parity import RiskParityEngine
from iso_financial_mvp.simulation_engines.cvar import CVaRCalculator
from iso_financial_mvp.simulation_engines.bayesian_ranker import BayesianScenarioRanker
from iso_financial_mvp.data_sources.polygon_client import PolygonDataSource
from iso_financial_mvp.llm_interface.ticker_identifier import TickerIdentifier
from iso_financial_mvp.report_generator.report_service import ReportService

# Setup logger
logger = logging.getLogger(__name__)

# Initialize components
query_parser = QueryParser()
monte_carlo = MonteCarloSimulator(num_simulations=1000)
scenario_engine = ScenarioAnalysisEngine(base_simulator=monte_carlo)
risk_parity = RiskParityEngine()
cvar_calculator = CVaRCalculator(confidence_level=0.95)
bayesian_ranker = BayesianScenarioRanker()
data_source = PolygonDataSource()
ticker_identifier = TickerIdentifier(data_source)
report_service = ReportService()

# Initialize scenario ranker with common scenarios
bayesian_ranker.initialize_common_scenarios()

# Create blueprint
report_api = Blueprint('report_api', __name__)

@report_api.route('/generate-report', methods=['POST'])
def generate_report():
    """
    API endpoint to generate a report (free or premium)
    """
    try:
        # Get data from request
        data = request.json
        query = data.get('query', '')
        user_profile = data.get('user_profile', 'novice')
        portfolio = data.get('portfolio', [])
        report_type = data.get('report_type', 'free')  # 'free' or 'premium'
        
        # Check if report generation is allowed
        if report_type == 'premium' and not data.get('premium_access', False):
            # Downgrade to free report if premium access not granted
            report_type = 'free'
            logger.warning("Downgraded to free report due to lack of premium access")
        
        # Parse the query
        query_data = query_parser.parse_query(query)
        
        # Identify assets in query if not provided in portfolio
        if not portfolio:
            identified_assets = ticker_identifier.identify_assets_in_query(query)
            
            if identified_assets:
                # Set equal weights
                weight = 1.0 / len(identified_assets)
                for asset in identified_assets:
                    asset['weight'] = weight
                
                portfolio = identified_assets
        
        # Enhance portfolio with additional data
        portfolio = ticker_identifier.enhance_portfolio_data(portfolio)
        
        # Add portfolio data to query
        query_data["portfolio"] = portfolio
        
        # Set initial investment if not provided
        if "initial_investment" not in query_data:
            query_data["initial_investment"] = 100000  # Default $100,000
        
        # Run simulations
        simulation_results = run_simulations(query_data, user_profile)
        
        # Generate report
        report_info = report_service.generate_report(
            query_data=query_data,
            simulation_results=simulation_results,
            user_profile=user_profile,
            report_type=report_type
        )
        
        # Return result
        return jsonify({
            "success": True,
            "report_path": report_info["report_path"],
            "report_level": report_info["report_level"],
            "query_data": query_data,
            "report_summary": report_info["report_data"].get("summary", "")
        })
    
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@report_api.route('/download-report/<path:filename>')
def download_report(filename):
    """
    Download a generated report
    """
    try:
        # Ensure the file exists
        if not os.path.exists(filename):
            return jsonify({"error": "Report not found"}), 404
        
        # Send the file
        return send_file(filename, as_attachment=True)
    
    except Exception as e:
        logger.error(f"Error downloading report: {str(e)}")
        return jsonify({"error": str(e)}), 500

@report_api.route('/compare-reports', methods=['POST'])
def compare_reports():
    """
    Generate both free and premium report samples for comparison
    """
    try:
        # Get data from request
        data = request.json
        query = data.get('query', '')
        user_profile = data.get('user_profile', 'novice')
        portfolio = data.get('portfolio', [])
        
        # Parse the query
        query_data = query_parser.parse_query(query)
        
        # Identify assets in query if not provided in portfolio
        if not portfolio:
            identified_assets = ticker_identifier.identify_assets_in_query(query)
            
            if identified_assets:
                # Set equal weights
                weight = 1.0 / len(identified_assets)
                for asset in identified_assets:
                    asset['weight'] = weight
                
                portfolio = identified_assets
        
        # Enhance portfolio with additional data
        portfolio = ticker_identifier.enhance_portfolio_data(portfolio)
        
        # Add portfolio data to query
        query_data["portfolio"] = portfolio
        
        # Set initial investment if not provided
        if "initial_investment" not in query_data:
            query_data["initial_investment"] = 100000
        
        # Run simulations
        simulation_results = run_simulations(query_data, user_profile)
        
        # Generate free report sample
        free_report_path = report_service.generate_free_report(
            query_data=query_data,
            simulation_results=simulation_results,
            user_profile=user_profile
        )
        
        # Generate premium report sample
        premium_report_path = report_service.generate_premium_report(
            query_data=query_data,
            simulation_results=simulation_results,
            user_profile=user_profile
        )
        
        # Return result
        return jsonify({
            "success": True,
            "free_report_path": free_report_path,
            "premium_report_path": premium_report_path,
            "query_data": query_data
        })
    
    except Exception as e:
        logger.error(f"Error comparing reports: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

def run_simulations(query_data, user_profile):
    """
    Run appropriate simulation models based on query type
    """
    query_type = query_data.get("query_type", "portfolio_analysis")
    results = {}
    
    # Create portfolio configuration
    portfolio_config = create_portfolio_config(query_data)
    
    # Monte Carlo simulation (used for most query types)
    if query_type in ["portfolio_analysis", "retirement_planning", "risk_assessment"]:
        mc_results = monte_carlo.simulate_portfolio(
            portfolio_config=portfolio_config,
            timeframe=query_data.get("timeframe", {"unit": "years", "value": 10}),
            risk_profile=query_data.get("risk_profile", "moderate")
        )
        results["monte_carlo"] = mc_results
    
    # Risk assessment specific models
    if query_type == "risk_assessment":
        # Calculate CVaR
        if "monte_carlo" in results:
            mc_paths = results["monte_carlo"]["simulation_paths"]
            initial_investment = portfolio_config.get("initial_investment", 100000)
            
            cvar_results = cvar_calculator.calculate_var_cvar(
                simulation_paths=mc_paths,
                initial_investment=initial_investment
            )
            
            # Add tail risk analysis
            tail_risk = cvar_calculator.analyze_tail_risk(
                simulation_paths=mc_paths,
                initial_investment=initial_investment
            )
            
            # Add stress test
            stress_test = cvar_calculator.stress_test_portfolio(
                simulation_paths=mc_paths,
                initial_investment=initial_investment
            )
            
            results["cvar"] = {
                **cvar_results,
                "tail_risk": tail_risk,
                "stress_scenarios": stress_test,
                "initial_investment": initial_investment
            }
        
        # Calculate risk parity allocation
        # Extract volatilities from portfolio assets
        assets_with_vol = []
        for asset in portfolio_config.get("assets", []):
            asset_type = asset.get("type", "")
            asset_name = asset.get("name", "")
            
            # Get volatility from asset data if available
            volatility = asset.get("annual_volatility", 0.2)  # Default 20%
            
            assets_with_vol.append({
                "name": asset_name,
                "type": asset_type,
                "volatility": volatility
            })
        
        # Get current weights
        num_assets = len(assets_with_vol)
        if num_assets > 0:
            current_weights = [asset.get("weight", 1.0 / num_assets) for asset in portfolio_config.get("assets", [])]
            
            # Calculate risk parity allocation
            rp_results = risk_parity.calculate_risk_parity_allocation(
                portfolio_assets=assets_with_vol
            )
            
            # Generate trade recommendations
            trades = risk_parity.recommend_trades(
                current_weights=current_weights,
                target_weights=rp_results["weights"],
                asset_details=rp_results["asset_details"],
                total_portfolio_value=portfolio_config.get("initial_investment", 100000)
            )
            
            results["risk_parity"] = {
                **rp_results,
                "trades": trades
            }
    
    # Scenario analysis specific models
    if query_type == "scenario_analysis":
        # Extract events from query
        events = query_data.get("events", [])
        
        # If no specific events found, use a default scenario
        if not events:
            scenario = scenario_engine.get_prebuilt_scenario("crypto_winter")
        else:
            # Create scenario from events
            event = events[0]  # Take first event
            
            if event.get("type") == "price_change":
                # Extract asset information
                asset_name = event.get("asset", "ethereum")
                asset_type = "crypto"  # Default to crypto
                
                # Find matching asset type from portfolio
                for asset in portfolio_config.get("assets", []):
                    if asset.get("name") == asset_name:
                        asset_type = asset.get("type", "crypto")
                        break
                
                # Create price change scenario
                scenario = scenario_engine.create_price_change_scenario(
                    asset_type=asset_type,
                    asset_name=asset_name,
                    price_change_pct=event.get("change", -0.3)
                )
            
            elif event.get("type") == "market_crash":
                scenario = scenario_engine.get_prebuilt_scenario("recession")
            
            elif event.get("type") == "regulatory_event":
                scenario = scenario_engine.get_prebuilt_scenario("defi_regulation")
            
            else:
                # Default scenario
                scenario = scenario_engine.get_prebuilt_scenario("crypto_winter")
        
        # Run scenario analysis
        scenario_results = scenario_engine.run_scenario_analysis(
            base_portfolio_config=portfolio_config,
            timeframe=query_data.get("timeframe", {"unit": "years", "value": 10}),
            risk_profile=query_data.get("risk_profile", "moderate"),
            scenario=scenario
        )
        
        results["scenario"] = scenario_results
        
        # Run Bayesian scenario ranking
        # Get portfolio composition for relevance calculation
        portfolio_composition = {}
        for asset in portfolio_config.get("assets", []):
            asset_type = asset.get("type", "")
            
            if asset_type in portfolio_composition:
                portfolio_composition[asset_type] += asset.get("weight", 1.0 / len(portfolio_config.get("assets", [])))
            else:
                portfolio_composition[asset_type] = asset.get("weight", 1.0 / len(portfolio_config.get("assets", [])))
        
        # Rank scenarios
        ranked_scenarios = bayesian_ranker.rank_scenarios(
            portfolio_composition=portfolio_composition
        )
        
        results["bayesian"] = {
            "ranked_scenarios": ranked_scenarios
        }
    
    return results

def create_portfolio_config(query_data):
    """
    Create portfolio configuration from query data
    """
    # Get assets from query data
    assets = query_data.get("assets", [])
    
    # If portfolio data is provided, use that instead
    if "portfolio" in query_data and query_data["portfolio"]:
        assets = query_data["portfolio"]
    
    # If no assets specified, use a default portfolio
    if not assets:
        assets = [
            {"type": "crypto", "name": "ethereum", "weight": 0.3},
            {"type": "crypto", "name": "bitcoin", "weight": 0.2},
            {"type": "traditional", "name": "us_stocks", "weight": 0.3},
            {"type": "traditional", "name": "bonds", "weight": 0.2}
        ]
    
    # Ensure weights sum to 1
    total_weight = sum(asset.get("weight", 0) for asset in assets)
    
    # If weights aren't specified or don't sum to 1, normalize or assign equal weights
    if abs(total_weight - 1.0) > 0.01 or total_weight == 0:
        if total_weight > 0:
            # Normalize existing weights
            for asset in assets:
                if "weight" in asset:
                    asset["weight"] = asset["weight"] / total_weight
        else:
            # Assign equal weights
            equal_weight = 1.0 / len(assets)
            for asset in assets:
                asset["weight"] = equal_weight
    
    # Get initial investment (default $100,000)
    initial_investment = query_data.get("initial_investment", 100000)
    
    return {
        "assets": assets,
        "initial_investment": initial_investment
    }