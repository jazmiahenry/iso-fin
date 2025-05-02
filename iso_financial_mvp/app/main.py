import os
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Query, Body
from pydantic import BaseModel, Field

from iso_financial_mvp.llm_interface.query_parser import QueryParser
from iso_financial_mvp.simulation_engines.monte_carlo import MonteCarloSimulator
from iso_financial_mvp.simulation_engines.scenario_analysis import ScenarioAnalysisEngine
from iso_financial_mvp.simulation_engines.risk_parity import RiskParityEngine
from iso_financial_mvp.simulation_engines.cvar import CVaRCalculator
from iso_financial_mvp.simulation_engines.bayesian_ranker import BayesianScenarioRanker
from iso_financial_mvp.report_generator.formatter import ReportFormatter
from iso_financial_mvp.gpt_agent import FinancialGPTAgent

# Create FastAPI app
app = FastAPI(
    title="ISO Financial MVP API",
    description="API for financial portfolio analysis and simulation",
    version="0.1.0"
)

# Initialize components
query_parser = QueryParser()
monte_carlo = MonteCarloSimulator(num_simulations=1000)
scenario_engine = ScenarioAnalysisEngine(base_simulator=monte_carlo)
risk_parity = RiskParityEngine()
cvar_calculator = CVaRCalculator(confidence_level=0.95)
bayesian_ranker = BayesianScenarioRanker()
report_formatter = ReportFormatter()

# Initialize Financial Educator agent with direct model reference
try:
    # Using the finetuned model directly
    model_name = "ft:gpt-4.1-2025-04-14:isoai:iso-financial-advisor-v1:BRuCm1OJ"
    print(f"Initializing Financial Educator with model: {model_name}")
    
    financial_educator = FinancialGPTAgent()
    print(f"Financial Educator initialized successfully")
except Exception as e:
    print(f"Warning: Could not initialize Financial Educator agent: {e}")
    financial_educator = None

# Initialize scenario ranker with common scenarios
bayesian_ranker.initialize_common_scenarios()

# Load asset metadata
@app.on_event("startup")
async def startup_event():
    global asset_metadata
    
    try:
        # Get path to assets_metadata.csv
        module_dir = os.path.dirname(os.path.dirname(__file__))
        asset_data_path = os.path.join(module_dir, "data", "assets_metadata.csv")
        
        # Load asset metadata
        asset_metadata = pd.read_csv(asset_data_path)
    except Exception as e:
        print(f"Warning: Could not load asset metadata: {e}")
        # Create empty DataFrame with required columns
        asset_metadata = pd.DataFrame(columns=[
            "asset_type", "asset_name", "annual_return", "annual_volatility", 
            "correlation_group", "liquidity"
        ])

# Define models
class NaturalLanguageQuery(BaseModel):
    query: str = Field(..., description="Natural language query from the user")
    user_profile: str = Field("novice", description="User expertise level (novice, intermediate, pro)")
    portfolio: Optional[List[Dict[str, Any]]] = Field(None, description="Optional portfolio composition")

class FinancialEducatorQuery(BaseModel):
    query: str = Field(..., description="Financial question to ask the Financial Educator")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context like portfolio data, market conditions, etc.")

class FinancialEducatorResponse(BaseModel):
    response: str = Field(..., description="Educational response from the Financial Educator")
    calculations: Optional[List[Dict[str, Any]]] = Field(None, description="Details of any calculations performed")

class SimulationResult(BaseModel):
    query_data: Dict[str, Any] = Field(..., description="Parsed query data")
    results: Dict[str, Any] = Field(..., description="Simulation results")
    report: Dict[str, Any] = Field(..., description="Formatted report")

# Define routes
@app.post("/financial-education", response_model=FinancialEducatorResponse)
async def get_financial_education(request: FinancialEducatorQuery):
    """
    Get educational insights from the Financial Educator
    
    This endpoint uses an AI model finetuned on financial data to provide
    educational content and analysis on financial topics and questions.
    """
    if financial_educator is None:
        raise HTTPException(
            status_code=503, 
            detail="Financial Educator is not available. Please try again later."
        )
    
    try:
        response = financial_educator.answer_question(request.query, request.context)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing educational query: {str(e)}"
        )

@app.post("/analyze", response_model=SimulationResult)
async def analyze_query(request: NaturalLanguageQuery):
    """
    Analyze a natural language financial query and generate a report
    """
    try:
        # Parse the query
        query_data = query_parser.parse_query(request.query)
        
        # Add portfolio data if provided
        if request.portfolio:
            query_data["portfolio"] = request.portfolio
        
        # Run appropriate models based on query type
        results = run_simulations(query_data, request.user_profile)
        
        # Format results into a report
        report = report_formatter.format_report(
            simulation_results=results,
            query_data=query_data,
            user_profile=request.user_profile
        )
        
        return {
            "query_data": query_data,
            "results": results,
            "report": report
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

def run_simulations(query_data: Dict[str, Any], user_profile: str) -> Dict[str, Any]:
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
                "stress_scenarios": stress_test
            }
        
        # Calculate risk parity allocation
        # Extract volatilities from portfolio assets
        assets_with_vol = []
        for asset in portfolio_config.get("assets", []):
            asset_type = asset.get("type", "")
            asset_name = asset.get("name", "")
            
            # Try to get volatility from asset metadata
            volatility = 0.2  # Default
            
            # Filter asset_metadata DataFrame
            matching_assets = asset_metadata[
                (asset_metadata["asset_type"] == asset_type) & 
                (asset_metadata["asset_name"] == asset_name)
            ]
            
            if not matching_assets.empty:
                volatility = matching_assets.iloc[0]["annual_volatility"]
            
            assets_with_vol.append({
                "name": asset_name,
                "type": asset_type,
                "volatility": volatility
            })
        
        # Get current weights (equal by default)
        num_assets = len(assets_with_vol)
        if num_assets > 0:
            current_weights = [1.0 / num_assets] * num_assets
            
            # Calculate risk parity allocation
            rp_results = risk_parity.calculate_risk_parity_allocation(
                portfolio_assets=assets_with_vol
            )
            
            # Generate trade recommendations
            trades = risk_parity.recommend_trades(
                current_weights=current_weights,
                target_weights=rp_results["weights"],
                asset_details=rp_results["asset_details"]
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
                portfolio_composition[asset_type] += 1
            else:
                portfolio_composition[asset_type] = 1
        
        # Normalize to get weights
        total_assets = sum(portfolio_composition.values())
        if total_assets > 0:
            for asset_type in portfolio_composition:
                portfolio_composition[asset_type] /= total_assets
        
        # Rank scenarios
        ranked_scenarios = bayesian_ranker.rank_scenarios(
            portfolio_composition=portfolio_composition
        )
        
        results["bayesian"] = {
            "ranked_scenarios": ranked_scenarios
        }
    
    return results

def create_portfolio_config(query_data: Dict[str, Any]) -> Dict[str, Any]:
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)