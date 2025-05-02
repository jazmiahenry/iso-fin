"""
Financial data tools for accessing historical data, portfolio simulations, and risk analysis.
These tools are designed to be used by the LLM agents.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Union, Optional, Tuple
from datetime import datetime, timedelta

# Import data sources and simulation engines
from iso_financial_mvp.data_sources.polygon_client import PolygonDataSource
from iso_financial_mvp.simulation_engines.monte_carlo import MonteCarloSimulator
from iso_financial_mvp.simulation_engines.cvar import CVaRCalculator
from iso_financial_mvp.simulation_engines.risk_parity import RiskParityEngine
from iso_financial_mvp.simulation_engines.scenario_analysis import ScenarioAnalysisEngine
from iso_financial_mvp.simulation_engines.bayesian_ranker import BayesianScenarioRanker

# Initialize data sources and engines
data_source = PolygonDataSource()
monte_carlo = MonteCarloSimulator(num_simulations=500)  # Use fewer simulations for faster response
cvar_calculator = CVaRCalculator()
risk_parity_engine = RiskParityEngine()
scenario_engine = ScenarioAnalysisEngine(base_simulator=monte_carlo)
bayesian_ranker = BayesianScenarioRanker()

# Initialize Bayesian ranker with common scenarios
bayesian_ranker.initialize_common_scenarios()

# ===== Historical Data Tools =====

def get_historical_data(tickers: List[str], days: int = 365) -> Dict[str, Any]:
    """
    Get historical price data for a list of ticker symbols
    
    Args:
        tickers: List of ticker symbols (e.g. ["SPY", "QQQ", "AGG"])
        days: Number of days of history to retrieve (default: 365)
        
    Returns:
        Dictionary with historical data summary and statistics
    """
    try:
        # Get historical data
        historical_data = data_source.get_historical_data(tickers, days=days)
        
        if historical_data.empty:
            return {
                "success": False,
                "error": "No historical data found for the provided tickers",
                "tickers": tickers
            }
        
        # Calculate returns
        returns = historical_data.pct_change().dropna()
        
        # Calculate statistics
        stats = {}
        for ticker in historical_data.columns:
            ticker_data = historical_data[ticker].dropna()
            ticker_returns = returns[ticker].dropna()
            
            if len(ticker_data) < 2 or len(ticker_returns) < 2:
                continue
                
            start_price = ticker_data.iloc[0]
            end_price = ticker_data.iloc[-1]
            total_return = (end_price / start_price) - 1
            annualized_return = (1 + total_return) ** (252 / len(ticker_data)) - 1
            
            stats[ticker] = {
                "start_price": float(start_price),
                "end_price": float(end_price),
                "total_return": float(total_return),
                "annualized_return": float(annualized_return),
                "annualized_volatility": float(ticker_returns.std() * (252 ** 0.5)),
                "sharpe_ratio": float(annualized_return / (ticker_returns.std() * (252 ** 0.5))) if ticker_returns.std() > 0 else 0,
                "data_points": len(ticker_data)
            }
        
        # Calculate correlation matrix if we have multiple tickers
        correlation_matrix = None
        if len(returns.columns) > 1:
            correlation_matrix = returns.corr().to_dict()
            
        return {
            "success": True,
            "tickers": tickers,
            "days": days,
            "stats": stats,
            "correlation_matrix": correlation_matrix,
            "data_length": len(historical_data)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "tickers": tickers
        }

def get_asset_info(asset_type: str, asset_name: str) -> Dict[str, Any]:
    """
    Get detailed information about an asset
    
    Args:
        asset_type: Type of asset (crypto, traditional, alternative)
        asset_name: Name of the asset
        
    Returns:
        Dictionary with asset information
    """
    try:
        # Get ticker
        ticker = data_source.get_ticker_for_asset(asset_type, asset_name)
        
        if not ticker:
            return {
                "success": False,
                "error": f"No ticker found for {asset_type}/{asset_name}",
                "asset_type": asset_type,
                "asset_name": asset_name
            }
        
        # Get current price
        current_price = data_source.get_current_price(asset_type, asset_name)
        
        # Get returns and volatility
        annual_return, annual_volatility = data_source.get_returns_and_volatility(
            asset_type, asset_name, days=365
        )
        
        return {
            "success": True,
            "asset_type": asset_type,
            "asset_name": asset_name,
            "ticker": ticker,
            "current_price": current_price,
            "annual_return": annual_return,
            "annual_volatility": annual_volatility
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "asset_type": asset_type,
            "asset_name": asset_name
        }

# ===== Portfolio Simulation Tools =====

def simulate_portfolio(
    portfolio: Dict[str, float], 
    initial_investment: float = 10000,
    years: int = 10,
    risk_profile: str = "moderate"
) -> Dict[str, Any]:
    """
    Run a Monte Carlo simulation for a portfolio
    
    Args:
        portfolio: Dictionary mapping ticker symbols to weights (must sum to 1)
        initial_investment: Initial investment amount (default: 10000)
        years: Number of years to simulate (default: 10)
        risk_profile: Risk profile (conservative, moderate, aggressive)
        
    Returns:
        Dictionary with simulation results
    """
    try:
        # Check portfolio weights
        total_weight = sum(portfolio.values())
        if abs(total_weight - 1.0) > 0.01:
            return {
                "success": False,
                "error": f"Portfolio weights must sum to 1.0 (current sum: {total_weight})",
                "portfolio": portfolio
            }
        
        # Format portfolio for simulator
        assets = []
        for ticker, weight in portfolio.items():
            # Determine asset type from ticker
            asset_type = "traditional"  # Default
            if ticker.startswith("X:"):
                asset_type = "crypto"
            
            assets.append({
                "name": ticker,
                "type": asset_type,
                "weight": weight
            })
            
        portfolio_config = {
            "assets": assets,
            "initial_investment": initial_investment
        }
        
        # Run simulation
        timeframe = {"unit": "years", "value": years}
        mc_results = monte_carlo.simulate_portfolio(
            portfolio_config=portfolio_config,
            timeframe=timeframe,
            risk_profile=risk_profile
        )
        
        # Extract key metrics
        final_values = mc_results["paths"][:, -1]
        
        metrics = {
            "mean_final_value": float(mc_results["mean_final"]),
            "median_final_value": float(mc_results["median_final"]),
            "min_final_value": float(np.min(final_values)),
            "max_final_value": float(np.max(final_values)),
            "percentiles": {
                "5": float(mc_results["percentiles"][5]),
                "25": float(mc_results["percentiles"][25]),
                "50": float(mc_results["percentiles"][50]),
                "75": float(mc_results["percentiles"][75]),
                "95": float(mc_results["percentiles"][95])
            },
            "probability_positive": float(mc_results["prob_positive"]),
            "expected_return": float(mc_results["expected_return"])
        }
        
        return {
            "success": True,
            "portfolio": portfolio,
            "initial_investment": initial_investment,
            "years": years,
            "risk_profile": risk_profile,
            "metrics": metrics
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "portfolio": portfolio
        }

# ===== Risk Analysis Tools =====

def analyze_portfolio_risk(
    portfolio: Dict[str, float], 
    initial_investment: float = 10000,
    years: int = 10,
    risk_profile: str = "moderate"
) -> Dict[str, Any]:
    """
    Analyze the risk of a portfolio
    
    Args:
        portfolio: Dictionary mapping ticker symbols to weights (must sum to 1)
        initial_investment: Initial investment amount (default: 10000)
        years: Number of years to simulate (default: 10)
        risk_profile: Risk profile (conservative, moderate, aggressive)
        
    Returns:
        Dictionary with risk analysis results
    """
    try:
        # First run the simulation
        sim_result = simulate_portfolio(
            portfolio=portfolio,
            initial_investment=initial_investment,
            years=years,
            risk_profile=risk_profile
        )
        
        if not sim_result["success"]:
            return sim_result
            
        # Format portfolio for simulator (same as in simulate_portfolio)
        assets = []
        for ticker, weight in portfolio.items():
            # Determine asset type from ticker
            asset_type = "traditional"  # Default
            if ticker.startswith("X:"):
                asset_type = "crypto"
            
            assets.append({
                "name": ticker,
                "type": asset_type,
                "weight": weight
            })
            
        portfolio_config = {
            "assets": assets,
            "initial_investment": initial_investment
        }
        
        # Run simulation again to get the paths
        timeframe = {"unit": "years", "value": years}
        mc_results = monte_carlo.simulate_portfolio(
            portfolio_config=portfolio_config,
            timeframe=timeframe,
            risk_profile=risk_profile
        )
        
        # Get simulation paths
        simulation_paths = mc_results["paths"]
        
        # Calculate VaR and CVaR
        cvar_results = cvar_calculator.calculate_var_cvar(
            simulation_paths=simulation_paths,
            initial_investment=initial_investment
        )
        
        # Calculate CDaR
        cdar_results = cvar_calculator._calculate_cdar(
            simulation_paths=simulation_paths,
            initial_investment=initial_investment
        )
        
        # Create risk metrics
        risk_metrics = {
            "var_95": float(cvar_results["var_amount"]),
            "cvar_95": float(cvar_results["cvar_amount"]),
            "var_99": float(cvar_results["comparison"]["var_99"]),
            "cvar_99": float(cvar_results["comparison"]["cvar_99"]),
            "worst_loss": float(cvar_results["worst_loss"]),
            "max_drawdown": float(cdar_results["maximum_drawdown"]["worst"]),
            "probability_of_loss": 1.0 - float(mc_results["prob_positive"])
        }
        
        return {
            "success": True,
            "portfolio": portfolio,
            "initial_investment": initial_investment,
            "years": years,
            "risk_profile": risk_profile,
            "risk_metrics": risk_metrics
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "portfolio": portfolio
        }

# ===== Portfolio Optimization Tools =====

def optimize_portfolio(portfolio: Dict[str, float]) -> Dict[str, Any]:
    """
    Optimize a portfolio using risk parity principles
    
    Args:
        portfolio: Dictionary mapping ticker symbols to weights (used for asset selection, not weights)
        
    Returns:
        Dictionary with optimized portfolio weights
    """
    try:
        # Format portfolio for optimizer
        portfolio_assets = []
        for ticker, weight in portfolio.items():
            # Determine volatility based on ticker
            if ticker.startswith("X:"):  # Crypto
                volatility = 0.70
            elif ticker in ["SPY", "QQQ", "VTI", "VOO", "IVV"]:  # Major ETFs
                volatility = 0.15
            elif ticker in ["AGG", "BND", "TLT", "IEF", "SHY"]:  # Bond ETFs
                volatility = 0.05
            elif ticker in ["GLD", "IAU", "SLV"]:  # Commodity ETFs
                volatility = 0.18
            else:
                volatility = 0.20  # Default
                
            # Add asset
            portfolio_assets.append({
                "name": ticker,
                "type": "stock" if not ticker.startswith("X:") else "crypto",
                "volatility": volatility,
                "weight": weight
            })
        
        # Run optimization
        try:
            optimization_results = risk_parity_engine.calculate_risk_parity_allocation(portfolio_assets)
        except Exception as optimizer_error:
            # Fallback to approximation method
            # Extract volatilities for approximation
            vols = np.array([asset["volatility"] for asset in portfolio_assets])
            
            # Generate identity correlation matrix
            n_assets = len(portfolio_assets)
            corr_matrix = np.eye(n_assets)
            
            # Add some default correlation
            for i in range(n_assets):
                for j in range(i+1, n_assets):
                    corr_matrix[i, j] = 0.3
                    corr_matrix[j, i] = 0.3
            
            # Calculate covariance matrix
            cov_matrix = np.diag(vols) @ corr_matrix @ np.diag(vols)
            
            # Use equal risk budget
            risk_budget = np.ones(n_assets) / n_assets
            
            # Use the approximation method
            weights = risk_parity_engine._approximate_risk_parity(cov_matrix, risk_budget)
            
            # Format results
            asset_details = []
            for i, asset in enumerate(portfolio_assets):
                asset_details.append({
                    "name": asset["name"],
                    "type": asset.get("type", "stock"),
                    "weight": float(weights[i]),
                    "volatility": float(vols[i]),
                    "risk_contribution": 1.0/n_assets,
                    "risk_contribution_pct": 1.0/n_assets
                })
            
            # Create optimization results
            optimization_results = {
                "weights": weights.tolist(),
                "portfolio_volatility": float(np.sqrt(weights.T @ cov_matrix @ weights)),
                "diversification_ratio": 1.0,
                "risk_contribution": [1.0/n_assets] * n_assets,
                "asset_details": asset_details
            }
        
        # Format results for return
        optimal_weights = {}
        for asset in optimization_results["asset_details"]:
            optimal_weights[asset["name"]] = asset["weight"]
        
        # Create return value
        return {
            "success": True,
            "original_portfolio": portfolio,
            "optimized_portfolio": optimal_weights,
            "portfolio_volatility": float(optimization_results["portfolio_volatility"]),
            "diversification_ratio": float(optimization_results.get("diversification_ratio", 1.0))
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "portfolio": portfolio
        }

# ===== Scenario Analysis Tools =====

def analyze_scenario(
    portfolio: Dict[str, float], 
    initial_investment: float = 10000,
    years: int = 10,
    scenario_name: str = "recession"
) -> Dict[str, Any]:
    """
    Analyze how a portfolio would perform under a specific scenario
    
    Args:
        portfolio: Dictionary mapping ticker symbols to weights (must sum to 1)
        initial_investment: Initial investment amount (default: 10000)
        years: Number of years to simulate (default: 10)
        scenario_name: Name of the scenario (recession, inflation, market_crash, tech_boom, recovery)
        
    Returns:
        Dictionary with scenario analysis results
    """
    try:
        # Format portfolio for simulator
        assets = []
        for ticker, weight in portfolio.items():
            # Determine asset type from ticker
            asset_type = "traditional"  # Default
            if ticker.startswith("X:"):
                asset_type = "crypto"
            
            assets.append({
                "name": ticker,
                "type": asset_type,
                "weight": weight
            })
            
        portfolio_config = {
            "assets": assets,
            "initial_investment": initial_investment
        }
        
        # Get the scenario
        scenario = scenario_engine.get_prebuilt_scenario(scenario_name)
        if not scenario:
            return {
                "success": False,
                "error": f"Scenario '{scenario_name}' not found",
                "available_scenarios": ["recession", "inflation", "market_crash", "tech_boom", "recovery"]
            }
        
        # Run scenario analysis
        timeframe = {"unit": "years", "value": years}
        risk_profile = "moderate"  # Default
        scenario_results = scenario_engine.run_scenario_analysis(
            base_portfolio_config=portfolio_config,
            timeframe=timeframe,
            risk_profile=risk_profile,
            scenario=scenario
        )
        
        # Create comparison metrics
        comparison = scenario_results.get("comparison", {})
        mean_change = comparison.get("mean_change", {})
        
        return {
            "success": True,
            "portfolio": portfolio,
            "initial_investment": initial_investment,
            "years": years,
            "scenario": scenario_name,
            "baseline_final_value": float(scenario_results.get("baseline", {}).get("final_value", 0)),
            "scenario_final_value": float(scenario_results.get("scenario", {}).get("final_value", 0)),
            "absolute_change": float(mean_change.get("absolute", 0)),
            "percentage_change": float(mean_change.get("percentage", 0)),
            "probability_of_loss": float(comparison.get("probability_of_loss", {}).get("difference", 0)),
            "recovery_probability": float(comparison.get("recovery_analysis", {}).get("paths_that_recover_pct", 0)),
            "scenario_description": scenario.get("description", "")
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "portfolio": portfolio
        }

# ===== Bayesian Scenario Analysis Tools =====

def rank_scenarios_for_portfolio(portfolio: Dict[str, float]) -> Dict[str, Any]:
    """
    Rank potential scenarios by their likelihood and impact on a portfolio
    
    Args:
        portfolio: Dictionary mapping ticker symbols to weights
        
    Returns:
        Dictionary with scenario rankings
    """
    try:
        # Calculate scenario likelihoods and impacts
        scenario_rankings = []
        
        # Get all scenarios from the bayesian ranker
        scenarios = bayesian_ranker.get_all_scenarios()
        
        for scenario_id, scenario in scenarios.items():
            # Calculate the scenario probability
            probability = bayesian_ranker.get_scenario_probability(scenario_id)
            
            # Calculate the estimated impact on portfolio
            impact = bayesian_ranker.estimate_portfolio_impact(
                scenario_id=scenario_id,
                portfolio=portfolio
            )
            
            # Add to rankings
            scenario_rankings.append({
                "scenario_id": scenario_id,
                "name": scenario.get("name", "Unknown"),
                "description": scenario.get("description", ""),
                "probability": float(probability),
                "impact": float(impact),
                "severity": float(probability * abs(impact)),
                "category": scenario.get("category", "general")
            })
        
        # Sort by severity (probability * impact)
        scenario_rankings.sort(key=lambda x: x["severity"], reverse=True)
        
        return {
            "success": True,
            "portfolio": portfolio,
            "scenario_rankings": scenario_rankings,
            "top_scenario": scenario_rankings[0] if scenario_rankings else None
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "portfolio": portfolio
        }

def update_scenario_probabilities(scenario_updates: Dict[str, float]) -> Dict[str, Any]:
    """
    Update the probabilities of scenarios based on new information
    
    Args:
        scenario_updates: Dictionary mapping scenario IDs to new probability values
        
    Returns:
        Dictionary with updated scenario probabilities
    """
    try:
        # Update each scenario probability
        updated_scenarios = []
        
        for scenario_id, probability in scenario_updates.items():
            # Check if probability is valid
            if probability < 0 or probability > 1:
                return {
                    "success": False,
                    "error": f"Invalid probability for scenario {scenario_id}: {probability}. Must be between 0 and 1.",
                    "scenario_updates": scenario_updates
                }
            
            # Update the probability
            bayesian_ranker.update_scenario_probability(scenario_id, probability)
            
            # Get the scenario details
            scenario = bayesian_ranker.get_scenario(scenario_id)
            
            # Add to updated scenarios
            updated_scenarios.append({
                "scenario_id": scenario_id,
                "name": scenario.get("name", "Unknown"),
                "new_probability": probability
            })
        
        return {
            "success": True,
            "updated_scenarios": updated_scenarios
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "scenario_updates": scenario_updates
        }

def get_available_scenarios() -> Dict[str, Any]:
    """
    Get all available scenarios from the Bayesian ranker
    
    Returns:
        Dictionary with available scenarios
    """
    try:
        # Get all scenarios
        scenarios = bayesian_ranker.get_all_scenarios()
        
        # Format for return
        scenario_list = []
        for scenario_id, scenario in scenarios.items():
            scenario_list.append({
                "scenario_id": scenario_id,
                "name": scenario.get("name", "Unknown"),
                "description": scenario.get("description", ""),
                "probability": float(bayesian_ranker.get_scenario_probability(scenario_id)),
                "category": scenario.get("category", "general")
            })
        
        # Group by category
        categories = {}
        for scenario in scenario_list:
            category = scenario["category"]
            if category not in categories:
                categories[category] = []
            categories[category].append(scenario)
        
        return {
            "success": True,
            "scenarios": scenario_list,
            "categories": categories
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }