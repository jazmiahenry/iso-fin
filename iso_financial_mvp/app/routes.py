from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field

from iso_financial_mvp.llm_interface.query_parser import QueryParser
from iso_financial_mvp.simulation_engines.monte_carlo import MonteCarloSimulator
from iso_financial_mvp.simulation_engines.scenario_analysis import ScenarioAnalysisEngine
from iso_financial_mvp.simulation_engines.risk_parity import RiskParityEngine
from iso_financial_mvp.simulation_engines.cvar import CVaRCalculator
from iso_financial_mvp.simulation_engines.bayesian_ranker import BayesianScenarioRanker
from iso_financial_mvp.report_generator.formatter import ReportFormatter

# Create router
router = APIRouter(
    prefix="/api",
    tags=["financial-analysis"]
)

# Initialize components (these will be initialized in main.py)
query_parser = None
monte_carlo = None
scenario_engine = None
risk_parity = None
cvar_calculator = None
bayesian_ranker = None
report_formatter = None
asset_metadata = None

# Define models
class Asset(BaseModel):
    type: str = Field(..., description="Asset type (crypto, nft, real_estate, collectible, traditional)")
    name: str = Field(..., description="Asset name")
    weight: Optional[float] = Field(None, description="Portfolio weight (0-1)")
    amount: Optional[float] = Field(None, description="Amount in currency")
    
class Portfolio(BaseModel):
    assets: List[Asset] = Field(..., description="List of assets in the portfolio")
    initial_investment: Optional[float] = Field(100000, description="Total portfolio value")

class NaturalLanguageQuery(BaseModel):
    query: str = Field(..., description="Natural language query from the user")
    user_profile: str = Field("novice", description="User expertise level (novice, intermediate, pro)")
    portfolio: Optional[Portfolio] = Field(None, description="Optional portfolio composition")

class QueryResponse(BaseModel):
    query_type: str = Field(..., description="Type of query detected")
    timeframe: Dict[str, Any] = Field(..., description="Timeframe for analysis")
    assets: List[Dict[str, Any]] = Field(..., description="Assets detected or provided")
    events: List[Dict[str, Any]] = Field(..., description="Events or scenarios detected")
    risk_profile: str = Field(..., description="Risk profile detected")
    original_query: str = Field(..., description="Original query text")

class SimulationRequest(BaseModel):
    query_data: QueryResponse = Field(..., description="Parsed query data")
    portfolio: Optional[Portfolio] = Field(None, description="Optional portfolio composition")
    user_profile: str = Field("novice", description="User expertise level")

class SimulationResult(BaseModel):
    query_data: Dict[str, Any] = Field(..., description="Parsed query data")
    results: Dict[str, Any] = Field(..., description="Simulation results")
    report: Dict[str, Any] = Field(..., description="Formatted report")

# Define routes
@router.post("/query/parse", response_model=QueryResponse)
async def parse_query(request: NaturalLanguageQuery):
    """
    Parse a natural language financial query into structured data
    """
    global query_parser
    
    try:
        # Parse the query
        query_data = query_parser.parse_query(request.query)
        
        # Add portfolio data if provided
        if request.portfolio:
            # Convert to dict format expected by internal systems
            portfolio_assets = [
                {
                    "type": asset.type,
                    "name": asset.name,
                    "weight": asset.weight if asset.weight is not None else None,
                    "amount": asset.amount if asset.amount is not None else None
                }
                for asset in request.portfolio.assets
            ]
            query_data["portfolio"] = portfolio_assets
            query_data["initial_investment"] = request.portfolio.initial_investment
        
        return query_data
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error parsing query: {str(e)}")

@router.post("/simulate", response_model=SimulationResult)
async def run_simulation(request: SimulationRequest):
    """
    Run simulation models based on parsed query data
    """
    try:
        # Get query data
        query_data = request.query_data
        
        # Add portfolio data if provided
        if request.portfolio:
            portfolio_assets = [
                {
                    "type": asset.type,
                    "name": asset.name,
                    "weight": asset.weight if asset.weight is not None else None,
                    "amount": asset.amount if asset.amount is not None else None
                }
                for asset in request.portfolio.assets
            ]
            query_data["portfolio"] = portfolio_assets
            query_data["initial_investment"] = request.portfolio.initial_investment
        
        # Run simulations based on query type
        from iso_financial_mvp.app.main import run_simulations
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
        raise HTTPException(status_code=500, detail=f"Error running simulation: {str(e)}")

@router.get("/scenarios")
async def list_scenarios():
    """
    List available prebuilt scenarios
    """
    global scenario_engine, bayesian_ranker
    
    try:
        # Get scenarios from Bayesian ranker
        scenarios = []
        
        for scenario_id in bayesian_ranker.scenarios:
            scenario_details = bayesian_ranker.get_scenario_details(scenario_id)
            scenarios.append({
                "id": scenario_id,
                "name": scenario_details["name"],
                "description": scenario_details["description"],
                "category": scenario_details["category"],
                "probability": scenario_details["probability"]
            })
        
        return {
            "scenarios": scenarios
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing scenarios: {str(e)}")

@router.get("/assets")
async def list_assets():
    """
    List available assets from metadata
    """
    global asset_metadata
    
    try:
        # Convert asset metadata to list of dictionaries
        if asset_metadata is not None:
            assets = asset_metadata.to_dict(orient="records")
        else:
            assets = []
        
        return {
            "assets": assets
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing assets: {str(e)}")

@router.post("/report/generate")
async def generate_report(results: SimulationResult):
    """
    Generate a formatted report from simulation results
    """
    global report_formatter
    
    try:
        # Format results into a report
        report = report_formatter.format_report(
            simulation_results=results.results,
            query_data=results.query_data,
            user_profile=results.query_data.get("user_profile", "novice")
        )
        
        return {
            "report": report
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")