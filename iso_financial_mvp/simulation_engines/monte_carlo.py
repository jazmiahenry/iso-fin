import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional

class MonteCarloSimulator:
    def __init__(self, num_simulations: int = 1000, random_seed: Optional[int] = None):
        """
        Initialize the Monte Carlo simulator
        
        Args:
            num_simulations: Number of simulation paths to generate
            random_seed: Optional seed for reproducibility
        """
        self.num_simulations = num_simulations
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def simulate_portfolio(self, 
                          portfolio_config: Dict[str, Any],
                          timeframe: Dict[str, Any],
                          risk_profile: str) -> Dict[str, Any]:
        """
        Run Monte Carlo simulations for a portfolio
        
        Args:
            portfolio_config: Asset allocation and parameters
            timeframe: Simulation timeframe (unit and value)
            risk_profile: User's risk profile
            
        Returns:
            Simulation results including paths and statistics
        """
        # Extract parameters
        assets = portfolio_config.get("assets", [])
        initial_investment = portfolio_config.get("initial_investment", 100000)
        
        # Set up time parameters
        unit = timeframe.get("unit", "years")
        time_horizon = timeframe.get("value", 10)
        
        # Convert months to years if needed
        if unit == "months":
            time_horizon_years = time_horizon / 12
        else:
            time_horizon_years = time_horizon
        
        # Number of steps (monthly)
        num_steps = int(time_horizon_years * 12)
        
        # Asset return and volatility assumptions based on risk profile
        return_vol_params = self._get_asset_parameters(assets, risk_profile)
        
        # Generate correlation matrix
        correlation_matrix = self._generate_correlation_matrix(assets)
        
        # Run simulation
        simulation_paths = self._generate_paths(
            initial_investment=initial_investment,
            return_vol_params=return_vol_params,
            correlation_matrix=correlation_matrix,
            num_steps=num_steps
        )
        
        # Calculate statistics
        stats = self._calculate_statistics(simulation_paths)
        
        return {
            "simulation_paths": simulation_paths,
            "statistics": stats,
            "params": {
                "initial_investment": initial_investment,
                "time_horizon": time_horizon,
                "time_unit": unit,
                "num_simulations": self.num_simulations,
                "assets": assets,
                "risk_profile": risk_profile
            }
        }
    
    def _get_asset_parameters(self, assets: List[Dict[str, Any]], risk_profile: str) -> List[Dict[str, Any]]:
        """Get return and volatility parameters for each asset based on type and risk profile"""
        params = []
        
        # These would come from a database or model in production
        # Using simplified assumptions here
        
        risk_multipliers = {
            "conservative": 0.8,
            "moderate": 1.0,
            "aggressive": 1.2
        }
        
        multiplier = risk_multipliers.get(risk_profile, 1.0)
        
        for asset in assets:
            asset_type = asset.get("type", "")
            asset_name = asset.get("name", "")
            
            # Default parameters
            annual_return = 0.08 * multiplier
            annual_volatility = 0.15 * multiplier
            
            # Adjust based on asset type
            if asset_type == "crypto":
                annual_return = 0.20 * multiplier
                annual_volatility = 0.60 * multiplier
                
                # Adjust for specific cryptos
                if asset_name == "ethereum":
                    annual_return = 0.25 * multiplier
                    annual_volatility = 0.70 * multiplier
                    
            elif asset_type == "nft":
                annual_return = 0.30 * multiplier
                annual_volatility = 0.80 * multiplier
                
            elif asset_type == "real_estate":
                annual_return = 0.06 * multiplier
                annual_volatility = 0.12 * multiplier
                
            elif asset_type == "collectible":
                annual_return = 0.10 * multiplier
                annual_volatility = 0.30 * multiplier
                
                if asset_name == "art":
                    annual_return = 0.08 * multiplier
                    annual_volatility = 0.25 * multiplier
                
            elif asset_type == "traditional":
                if asset_name == "stocks":
                    annual_return = 0.10 * multiplier
                    annual_volatility = 0.18 * multiplier
                elif asset_name == "bonds":
                    annual_return = 0.04 * multiplier
                    annual_volatility = 0.05 * multiplier
            
            # Convert annual parameters to monthly
            monthly_return = annual_return / 12
            monthly_volatility = annual_volatility / np.sqrt(12)
            
            params.append({
                "asset": f"{asset_type}_{asset_name}",
                "monthly_return": monthly_return,
                "monthly_volatility": monthly_volatility,
                "weight": asset.get("weight", 1.0 / len(assets))  # Equal weight by default
            })
        
        return params
    
    def _generate_correlation_matrix(self, assets: List[Dict[str, Any]]) -> np.ndarray:
        """Generate a correlation matrix for the assets"""
        n_assets = len(assets)
        
        # Start with identity matrix (no correlation)
        corr_matrix = np.eye(n_assets)
        
        # In production, this would use real correlation data
        # For now, use simplified assumptions
        
        # Map asset types to indices
        asset_types = [asset["type"] for asset in assets]
        asset_names = [asset["name"] for asset in assets]
        
        # Add correlations between assets
        for i in range(n_assets):
            for j in range(i+1, n_assets):
                # Base correlation
                correlation = 0.3
                
                # Adjust based on asset types
                if asset_types[i] == asset_types[j]:
                    # Same asset type
                    correlation = 0.7
                    
                    # If same specific asset, perfect correlation
                    if asset_names[i] == asset_names[j]:
                        correlation = 1.0
                
                # Traditional assets correlate differently with crypto
                if "traditional" in [asset_types[i], asset_types[j]] and "crypto" in [asset_types[i], asset_types[j]]:
                    correlation = 0.2
                
                # NFTs and collectibles might be more correlated
                if set([asset_types[i], asset_types[j]]) <= set(["nft", "collectible"]):
                    correlation = 0.5
                
                # Set the correlation (symmetric)
                corr_matrix[i, j] = correlation
                corr_matrix[j, i] = correlation
        
        return corr_matrix
    
    def _generate_paths(self, 
                       initial_investment: float,
                       return_vol_params: List[Dict[str, Any]],
                       correlation_matrix: np.ndarray,
                       num_steps: int) -> np.ndarray:
        """Generate Monte Carlo simulation paths"""
        n_assets = len(return_vol_params)
        
        # Extract parameters
        monthly_returns = np.array([param["monthly_return"] for param in return_vol_params])
        monthly_vols = np.array([param["monthly_volatility"] for param in return_vol_params])
        weights = np.array([param["weight"] for param in return_vol_params])
        
        # Normalize weights
        weights = weights / weights.sum()
        
        # Cholesky decomposition for correlated random numbers
        L = np.linalg.cholesky(correlation_matrix)
        
        # Initialize array for simulation results
        # Shape: (num_simulations, num_steps+1)
        # +1 for initial value
        all_paths = np.zeros((self.num_simulations, num_steps + 1))
        all_paths[:, 0] = initial_investment
        
        # Generate paths
        for t in range(1, num_steps + 1):
            # Generate correlated random numbers
            Z = np.random.standard_normal(size=(self.num_simulations, n_assets))
            correlated_Z = Z @ L.T
            
            # Calculate asset returns
            asset_returns = np.zeros((self.num_simulations, n_assets))
            for i in range(n_assets):
                asset_returns[:, i] = monthly_returns[i] + monthly_vols[i] * correlated_Z[:, i]
            
            # Calculate portfolio returns
            portfolio_returns = np.sum(asset_returns * weights, axis=1)
            
            # Update paths
            all_paths[:, t] = all_paths[:, t-1] * (1 + portfolio_returns)
        
        return all_paths
    
    def _calculate_statistics(self, paths: np.ndarray) -> Dict[str, Any]:
        """Calculate statistics from simulation paths"""
        # Final values (end of simulation)
        final_values = paths[:, -1]
        
        # Calculate statistics
        mean_final = np.mean(final_values)
        median_final = np.median(final_values)
        std_final = np.std(final_values)
        
        # Calculate percentiles for confidence intervals
        percentiles = [5, 25, 50, 75, 95]
        confidence_intervals = {}
        
        for p in percentiles:
            confidence_intervals[f"p{p}"] = np.percentile(final_values, p)
        
        # Calculate Value at Risk (VaR) at 95% confidence
        var_95 = np.percentile(final_values, 5)
        
        # Calculate Conditional Value at Risk (CVaR)
        cvar_95 = np.mean(final_values[final_values <= var_95])
        
        # Calculate maximum drawdown across all paths
        drawdowns = []
        for i in range(self.num_simulations):
            path = paths[i, :]
            peak = path[0]
            max_drawdown = 0
            
            for value in path[1:]:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
            
            drawdowns.append(max_drawdown)
        
        avg_max_drawdown = np.mean(drawdowns)
        max_max_drawdown = np.max(drawdowns)
        
        return {
            "mean": mean_final,
            "median": median_final,
            "std": std_final,
            "confidence_intervals": confidence_intervals,
            "var_95": var_95,
            "cvar_95": cvar_95,
            "avg_max_drawdown": avg_max_drawdown,
            "max_drawdown": max_max_drawdown
        }