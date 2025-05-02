import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False

class RiskParityEngine:
    def __init__(self, use_optimizer: bool = True):
        """
        Initialize the Risk Parity optimization engine
        
        Args:
            use_optimizer: Whether to use cvxpy optimization (if available) or approximation
        """
        self.use_optimizer = use_optimizer and HAS_CVXPY
    
    def calculate_risk_parity_allocation(self, 
                                       portfolio_assets: List[Dict[str, Any]],
                                       correlation_matrix: Optional[np.ndarray] = None,
                                       risk_budget: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Calculate risk parity asset allocation
        
        Args:
            portfolio_assets: List of assets with volatility information
            correlation_matrix: Optional correlation matrix between assets
            risk_budget: Optional target risk budget for each asset (defaults to equal)
            
        Returns:
            Dictionary with optimal weights and risk metrics
        """
        n_assets = len(portfolio_assets)
        
        # Extract volatilities
        vols = np.array([asset.get("volatility", asset.get("monthly_volatility", 0.1) * np.sqrt(12)) 
                        for asset in portfolio_assets])
        
        # Generate default correlation matrix if not provided
        if correlation_matrix is None:
            correlation_matrix = np.eye(n_assets)
            # Add some default correlation (0.3 between different assets)
            for i in range(n_assets):
                for j in range(i+1, n_assets):
                    correlation_matrix[i, j] = 0.3
                    correlation_matrix[j, i] = 0.3
        
        # Generate risk budget if not provided (equal by default)
        if risk_budget is None:
            risk_budget = np.ones(n_assets) / n_assets
        else:
            # Normalize risk budget to sum to 1
            risk_budget = np.array(risk_budget) / sum(risk_budget)
        
        # Calculate covariance matrix
        cov_matrix = np.diag(vols) @ correlation_matrix @ np.diag(vols)
        
        # Calculate risk parity weights
        if self.use_optimizer:
            weights = self._optimize_risk_parity(cov_matrix, risk_budget)
        else:
            weights = self._approximate_risk_parity(cov_matrix, risk_budget)
        
        # Calculate portfolio volatility with these weights
        portfolio_variance = weights.T @ cov_matrix @ weights
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Calculate risk contribution of each asset
        marginal_risk = (cov_matrix @ weights) / portfolio_volatility
        risk_contribution = weights * marginal_risk
        
        # Calculate diversification ratio
        weighted_vols = weights * vols
        sum_weighted_vols = np.sum(weighted_vols)
        diversification_ratio = sum_weighted_vols / portfolio_volatility
        
        # Prepare asset details for output
        asset_details = []
        for i, asset in enumerate(portfolio_assets):
            asset_details.append({
                "name": asset.get("name", f"Asset_{i}"),
                "type": asset.get("type", "unknown"),
                "weight": float(weights[i]),
                "volatility": float(vols[i]),
                "risk_contribution": float(risk_contribution[i]),
                "risk_contribution_pct": float(risk_contribution[i] / portfolio_volatility)
            })
        
        return {
            "weights": weights.tolist(),
            "portfolio_volatility": float(portfolio_volatility),
            "diversification_ratio": float(diversification_ratio),
            "risk_contribution": risk_contribution.tolist(),
            "asset_details": asset_details
        }
    
    def _optimize_risk_parity(self, 
                            cov_matrix: np.ndarray, 
                            risk_budget: np.ndarray) -> np.ndarray:
        """
        Optimize risk parity allocation using convex optimization
        
        Args:
            cov_matrix: Covariance matrix for the assets
            risk_budget: Target risk budget for each asset
            
        Returns:
            Optimal asset weights
        """
        n = len(risk_budget)
        
        if not HAS_CVXPY:
            return self._approximate_risk_parity(cov_matrix, risk_budget)
        
        # Define variables
        weights = cp.Variable(n)
        risk = cp.quad_form(weights, cov_matrix)
        
        # Define objective: minimize risk concentration
        risk_contributions = weights @ cov_matrix @ weights
        target_risk_contrib = risk * risk_budget
        objective = cp.sum_squares(cp.multiply(weights, cov_matrix @ weights) - target_risk_contrib)
        
        # Define constraints
        constraints = [
            cp.sum(weights) == 1,    # Weights sum to 1
            weights >= 0.01          # Non-negative weights with minimum allocation
        ]
        
        # Solve the problem
        problem = cp.Problem(cp.Minimize(objective), constraints)
        problem.solve()
        
        # Check if the problem was solved successfully
        if problem.status not in ["optimal", "optimal_inaccurate"]:
            return self._approximate_risk_parity(cov_matrix, risk_budget)
        
        return weights.value
    
    def _approximate_risk_parity(self,
                               cov_matrix: np.ndarray,
                               risk_budget: np.ndarray) -> np.ndarray:
        """
        Approximate risk parity using an iterative approach
        
        Args:
            cov_matrix: Covariance matrix for the assets
            risk_budget: Target risk budget for each asset
            
        Returns:
            Approximate risk parity weights
        """
        n = len(risk_budget)
        
        # Start with equal weighting
        weights = np.ones(n) / n
        
        # Iteratively adjust weights
        max_iter = 100
        for _ in range(max_iter):
            # Calculate portfolio variance and volatility
            portfolio_variance = weights.T @ cov_matrix @ weights
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            # Calculate risk contribution
            marginal_risk = (cov_matrix @ weights) / portfolio_volatility
            risk_contribution = weights * marginal_risk
            
            # Calculate risk contribution percentage
            risk_contrib_pct = risk_contribution / portfolio_volatility
            
            # Check if risk contributions match the budget
            rel_diff = risk_contrib_pct / risk_budget - 1
            if np.max(np.abs(rel_diff)) < 0.01:
                break
            
            # Update weights based on the difference
            weights = weights * (1 - 0.5 * rel_diff)
            
            # Normalize weights to sum to 1
            weights = weights / np.sum(weights)
        
        return weights
    
    def recommend_trades(self,
                        current_weights: List[float],
                        target_weights: List[float],
                        asset_details: List[Dict[str, Any]],
                        total_portfolio_value: float = 100000) -> Dict[str, Any]:
        """
        Generate trade recommendations to achieve target allocation
        
        Args:
            current_weights: Current portfolio weights
            target_weights: Target weights from risk parity
            asset_details: Details of each asset
            total_portfolio_value: Total portfolio value
            
        Returns:
            Trade recommendations
        """
        current_weights = np.array(current_weights)
        target_weights = np.array(target_weights)
        
        # Calculate weight differences
        weight_diff = target_weights - current_weights
        
        # Calculate dollar amounts
        current_amounts = current_weights * total_portfolio_value
        target_amounts = target_weights * total_portfolio_value
        amount_diff = target_amounts - current_amounts
        
        # Create trade recommendations
        trades = []
        for i, (diff, asset) in enumerate(zip(weight_diff, asset_details)):
            if abs(diff) > 0.005:  # Only recommend trades for significant changes (>0.5%)
                direction = "buy" if diff > 0 else "sell"
                trades.append({
                    "asset_name": asset.get("name", f"Asset_{i}"),
                    "asset_type": asset.get("type", "unknown"),
                    "direction": direction,
                    "weight_change": abs(float(diff)),
                    "amount": abs(float(amount_diff[i])),
                    "current_weight": float(current_weights[i]),
                    "target_weight": float(target_weights[i])
                })
        
        # Calculate potential impact on portfolio risk
        current_weights_array = np.array(current_weights)
        target_weights_array = np.array(target_weights)
        
        # Extract volatilities for a simple risk estimate
        vols = np.array([asset.get("volatility", 0.1) for asset in asset_details])
        
        # Simple estimate of risk reduction (ignoring correlations)
        current_weighted_vol = np.sum(current_weights_array * vols)
        target_weighted_vol = np.sum(target_weights_array * vols)
        
        risk_reduction_pct = (current_weighted_vol - target_weighted_vol) / current_weighted_vol if current_weighted_vol > 0 else 0
        
        return {
            "trades": trades,
            "total_trades_value": float(np.sum(np.abs(amount_diff))),
            "estimated_risk_reduction": float(risk_reduction_pct)
        }