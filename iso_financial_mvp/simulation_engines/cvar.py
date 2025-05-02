import numpy as np
from typing import Dict, Any, List, Tuple, Union

class CVaRCalculator:
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize the Conditional Value at Risk calculator
        
        Args:
            confidence_level: Confidence level for CVaR calculation (default 0.95)
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
    
    def calculate_var_cvar(self, 
                         simulation_paths: np.ndarray, 
                         initial_investment: float = None) -> Dict[str, Any]:
        """
        Calculate Value at Risk (VaR) and Conditional Value at Risk (CVaR)
        
        Args:
            simulation_paths: Monte Carlo simulation paths (num_simulations x num_periods)
            initial_investment: Optional initial investment amount
            
        Returns:
            Dictionary with VaR and CVaR metrics
        """
        # If initial investment is not provided, use the first value in the simulation
        if initial_investment is None:
            initial_investment = simulation_paths[0, 0]
        
        # Extract final values from simulation paths
        final_values = simulation_paths[:, -1]
        
        # Calculate returns
        returns = (final_values - initial_investment) / initial_investment
        
        # Calculate losses (negative returns)
        losses = -returns
        
        # Calculate VaR at the specified confidence level
        var = np.percentile(losses, self.confidence_level * 100)
        
        # Calculate CVaR (expected loss beyond VaR)
        cvar = np.mean(losses[losses >= var])
        
        # Calculate VaR and CVaR in monetary terms
        var_amount = var * initial_investment
        cvar_amount = cvar * initial_investment
        
        # Calculate additional risk metrics
        worst_loss = np.max(losses) * initial_investment
        probability_of_loss = np.mean(losses > 0)
        
        # Calculate VaR and CVaR at different confidence levels for comparison
        var_90 = np.percentile(losses, 90) * initial_investment
        cvar_90 = np.mean(losses[losses >= np.percentile(losses, 90)]) * initial_investment
        
        var_99 = np.percentile(losses, 99) * initial_investment
        cvar_99 = np.mean(losses[losses >= np.percentile(losses, 99)]) * initial_investment
        
        return {
            "var": float(var),
            "cvar": float(cvar),
            "var_amount": float(var_amount),
            "cvar_amount": float(cvar_amount),
            "worst_loss": float(worst_loss),
            "probability_of_loss": float(probability_of_loss),
            "confidence_level": self.confidence_level,
            "comparison": {
                "var_90": float(var_90),
                "cvar_90": float(cvar_90),
                "var_99": float(var_99),
                "cvar_99": float(cvar_99)
            }
        }
    
    def analyze_tail_risk(self, 
                         simulation_paths: np.ndarray,
                         initial_investment: float = None,
                         time_periods: List[int] = None) -> Dict[str, Any]:
        """
        Analyze tail risk across multiple time periods
        
        Args:
            simulation_paths: Monte Carlo simulation paths
            initial_investment: Initial investment amount (default: first value in paths)
            time_periods: List of time periods to analyze (default: quarter, half, full)
            
        Returns:
            Dictionary with tail risk metrics across time periods
        """
        num_simulations, num_periods = simulation_paths.shape
        
        # If initial investment is not provided, use the first value
        if initial_investment is None:
            initial_investment = simulation_paths[0, 0]
        
        # If time periods are not provided, use quarter, half, and full
        if time_periods is None:
            time_periods = [
                max(1, num_periods // 4),      # Quarter
                max(1, num_periods // 2),      # Half
                num_periods - 1                # Full (end)
            ]
        
        # Ensure time periods are valid
        time_periods = [min(t, num_periods - 1) for t in time_periods]
        
        # Calculate VaR and CVaR for each time period
        period_results = {}
        for period in time_periods:
            # Extract values at this time period
            period_values = simulation_paths[:, period]
            
            # Calculate returns
            period_returns = (period_values - initial_investment) / initial_investment
            
            # Calculate losses
            period_losses = -period_returns
            
            # Calculate VaR
            period_var = np.percentile(period_losses, self.confidence_level * 100)
            
            # Calculate CVaR
            period_cvar = np.mean(period_losses[period_losses >= period_var])
            
            # Convert to monetary values
            period_var_amount = period_var * initial_investment
            period_cvar_amount = period_cvar * initial_investment
            
            # Store results
            period_results[f"period_{period}"] = {
                "var": float(period_var),
                "cvar": float(period_cvar),
                "var_amount": float(period_var_amount),
                "cvar_amount": float(period_cvar_amount),
                "probability_of_loss": float(np.mean(period_losses > 0))
            }
        
        # Calculate conditional drawdown at risk (CDaR)
        cdar_results = self._calculate_cdar(simulation_paths, initial_investment)
        
        return {
            "period_risk": period_results,
            "drawdown_risk": cdar_results,
            "confidence_level": self.confidence_level
        }
    
    def _calculate_cdar(self, 
                      simulation_paths: np.ndarray,
                      initial_investment: float = None) -> Dict[str, Any]:
        """
        Calculate Conditional Drawdown at Risk (CDaR)
        
        Args:
            simulation_paths: Monte Carlo simulation paths
            initial_investment: Initial investment amount
            
        Returns:
            Dictionary with drawdown risk metrics
        """
        num_simulations, num_periods = simulation_paths.shape
        
        # Calculate maximum drawdown for each simulation path
        max_drawdowns = np.zeros(num_simulations)
        avg_drawdowns = np.zeros(num_simulations)
        drawdown_durations = np.zeros(num_simulations)
        
        for i in range(num_simulations):
            path = simulation_paths[i, :]
            
            # Calculate running maximum
            running_max = np.maximum.accumulate(path)
            
            # Calculate drawdowns
            drawdowns = (running_max - path) / running_max
            
            # Calculate maximum drawdown
            max_drawdowns[i] = np.max(drawdowns)
            
            # Calculate average drawdown
            avg_drawdowns[i] = np.mean(drawdowns)
            
            # Calculate drawdown duration
            in_drawdown = drawdowns > 0
            if np.any(in_drawdown):
                duration = 0
                max_duration = 0
                
                for j in range(1, len(in_drawdown)):
                    if in_drawdown[j]:
                        duration += 1
                    else:
                        max_duration = max(max_duration, duration)
                        duration = 0
                
                max_duration = max(max_duration, duration)
                drawdown_durations[i] = max_duration
        
        # Calculate CDaR
        dar = np.percentile(max_drawdowns, self.confidence_level * 100)
        cdar = np.mean(max_drawdowns[max_drawdowns >= dar])
        
        return {
            "maximum_drawdown": {
                "average": float(np.mean(max_drawdowns)),
                "median": float(np.median(max_drawdowns)),
                "worst": float(np.max(max_drawdowns))
            },
            "drawdown_at_risk": float(dar),
            "conditional_drawdown_at_risk": float(cdar),
            "average_drawdown": float(np.mean(avg_drawdowns)),
            "average_max_drawdown_duration": float(np.mean(drawdown_durations))
        }
    
    def stress_test_portfolio(self,
                            simulation_paths: np.ndarray,
                            stress_factors: List[float] = None,
                            initial_investment: float = None) -> Dict[str, Any]:
        """
        Perform stress testing by applying stress factors to simulation paths
        
        Args:
            simulation_paths: Monte Carlo simulation paths
            stress_factors: List of stress factors to apply (e.g., [0.9, 0.8, 0.7])
            initial_investment: Initial investment amount
            
        Returns:
            Dictionary with stress test results
        """
        if stress_factors is None:
            stress_factors = [0.9, 0.8, 0.7, 0.6, 0.5]  # 10% to 50% stress
        
        # If initial investment is not provided, use the first value
        if initial_investment is None:
            initial_investment = simulation_paths[0, 0]
        
        # Calculate baseline VaR and CVaR
        baseline = self.calculate_var_cvar(simulation_paths, initial_investment)
        
        # Apply stress factors and calculate VaR and CVaR for each
        stress_results = {}
        
        for factor in stress_factors:
            # Apply stress factor to all paths
            stressed_paths = simulation_paths.copy()
            stressed_paths[:, 1:] *= factor
            
            # Calculate VaR and CVaR for stressed paths
            stress_metrics = self.calculate_var_cvar(stressed_paths, initial_investment)
            
            # Store results
            stress_results[f"stress_{int((1-factor)*100)}pct"] = {
                "factor": factor,
                "var_amount": stress_metrics["var_amount"],
                "cvar_amount": stress_metrics["cvar_amount"],
                "var_change": stress_metrics["var_amount"] - baseline["var_amount"],
                "cvar_change": stress_metrics["cvar_amount"] - baseline["cvar_amount"],
                "probability_of_loss": stress_metrics["probability_of_loss"]
            }
        
        return {
            "baseline": {
                "var_amount": baseline["var_amount"],
                "cvar_amount": baseline["cvar_amount"]
            },
            "stress_scenarios": stress_results
        }