import numpy as np
from typing import Dict, Any, List, Optional
from iso_financial_mvp.simulation_engines.monte_carlo import MonteCarloSimulator

class ScenarioAnalysisEngine:
    def __init__(self, base_simulator: Optional[MonteCarloSimulator] = None):
        """
        Initialize the scenario analysis engine
        
        Args:
            base_simulator: Optional MonteCarloSimulator to use for scenarios
        """
        self.base_simulator = base_simulator or MonteCarloSimulator()
        self.prebuilt_scenarios = self._initialize_prebuilt_scenarios()
    
    def _initialize_prebuilt_scenarios(self) -> Dict[str, Dict[str, Any]]:
        """Initialize library of prebuilt scenarios"""
        return {
            "crypto_winter": {
                "name": "Crypto Winter",
                "description": "Extended bear market in crypto with significant price declines",
                "adjustments": {
                    "asset_returns": {
                        "crypto": -0.6,  # 60% decline in crypto returns
                        "nft": -0.7      # 70% decline in NFT returns
                    },
                    "asset_volatility": {
                        "crypto": 1.5,   # 50% increase in volatility
                        "nft": 1.7       # 70% increase in volatility
                    },
                    "correlation_adjustments": {
                        "crypto_crypto": 0.9,  # Higher correlation in crypto assets
                        "crypto_nft": 0.8      # Higher correlation between crypto and NFTs
                    }
                }
            },
            "regulatory_crackdown": {
                "name": "Regulatory Crackdown",
                "description": "Major regulatory actions against crypto and DeFi",
                "adjustments": {
                    "asset_returns": {
                        "crypto": -0.4,  # 40% decline in crypto returns
                        "nft": -0.3      # 30% decline in NFT returns
                    },
                    "asset_volatility": {
                        "crypto": 2.0,   # Double volatility
                        "nft": 1.5       # 50% increase in volatility
                    },
                    "correlation_adjustments": {
                        "crypto_traditional": -0.3  # Negative correlation with traditional assets
                    }
                }
            },
            "nft_liquidity_freeze": {
                "name": "NFT Liquidity Freeze",
                "description": "Severe reduction in NFT trading volume and liquidity",
                "adjustments": {
                    "asset_returns": {
                        "nft": -0.8      # 80% decline in NFT returns
                    },
                    "asset_volatility": {
                        "nft": 2.5       # 2.5x increase in volatility
                    }
                }
            },
            "eth_crash": {
                "name": "ETH Price Crash",
                "description": "Significant Ethereum price decline",
                "adjustments": {
                    "specific_assets": {
                        "crypto_ethereum": {
                            "return_adjustment": -0.5,  # 50% decrease
                            "volatility_adjustment": 1.8  # 80% increase in volatility
                        }
                    }
                }
            },
            "recession": {
                "name": "Economic Recession",
                "description": "Broad economic downturn affecting all asset classes",
                "adjustments": {
                    "asset_returns": {
                        "crypto": -0.3,
                        "nft": -0.4,
                        "real_estate": -0.2,
                        "collectible": -0.15,
                        "traditional": -0.25
                    },
                    "asset_volatility": {
                        "crypto": 1.4,
                        "nft": 1.5,
                        "real_estate": 1.3,
                        "collectible": 1.2,
                        "traditional": 1.5
                    },
                    "correlation_adjustments": {
                        "global": 0.8  # Higher correlation across all assets
                    }
                }
            },
            "inflation_surge": {
                "name": "Inflation Surge",
                "description": "Unexpected spike in inflation",
                "adjustments": {
                    "asset_returns": {
                        "crypto": 0.2,    # Potential hedge
                        "real_estate": 0.15,
                        "traditional": -0.1
                    },
                    "asset_volatility": {
                        "global": 1.3     # 30% increase across all assets
                    }
                }
            }
        }
    
    def run_scenario_analysis(self, 
                             base_portfolio_config: Dict[str, Any],
                             timeframe: Dict[str, Any],
                             risk_profile: str,
                             scenario: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a scenario analysis by modifying base assumptions
        
        Args:
            base_portfolio_config: Standard portfolio configuration
            timeframe: Simulation timeframe
            risk_profile: User's risk profile
            scenario: Scenario definition with adjustments
            
        Returns:
            Results comparing baseline and scenario outcomes
        """
        # First, run baseline simulation
        baseline_results = self.base_simulator.simulate_portfolio(
            portfolio_config=base_portfolio_config,
            timeframe=timeframe,
            risk_profile=risk_profile
        )
        
        # Create adjusted portfolio config based on scenario
        adjusted_portfolio = self._apply_scenario_adjustments(
            base_portfolio_config.copy(),
            scenario
        )
        
        # Run scenario simulation
        scenario_results = self.base_simulator.simulate_portfolio(
            portfolio_config=adjusted_portfolio,
            timeframe=timeframe,
            risk_profile=risk_profile
        )
        
        # Compare results
        comparison = self._compare_results(
            baseline_results,
            scenario_results,
            scenario.get('name', 'Custom Scenario')
        )
        
        return {
            "baseline": baseline_results,
            "scenario": scenario_results,
            "comparison": comparison,
            "scenario_details": scenario
        }
    
    def get_prebuilt_scenario(self, scenario_key: str) -> Optional[Dict[str, Any]]:
        """Get a prebuilt scenario by key"""
        return self.prebuilt_scenarios.get(scenario_key)
    
    def create_custom_scenario(self, 
                              name: str,
                              description: str,
                              adjustments: Dict[str, Any]) -> Dict[str, Any]:
        """Create a custom scenario with specific adjustments"""
        return {
            "name": name,
            "description": description,
            "adjustments": adjustments
        }
    
    def create_price_change_scenario(self,
                                   asset_type: str,
                                   asset_name: str,
                                   price_change_pct: float) -> Dict[str, Any]:
        """
        Create a scenario for a specific asset price change
        
        Args:
            asset_type: Type of asset (crypto, nft, etc.)
            asset_name: Name of the specific asset
            price_change_pct: Percentage change (-0.4 for 40% drop)
            
        Returns:
            Scenario definition
        """
        asset_key = f"{asset_type}_{asset_name}"
        
        # Volatility adjustment - higher for negative changes
        volatility_adjustment = 1.0
        if price_change_pct < 0:
            # More severe price drops cause more volatility
            volatility_adjustment = 1.0 + min(2.0, abs(price_change_pct) * 2)
        
        return {
            "name": f"{asset_name.title()} {int(price_change_pct*100)}% Price Change",
            "description": f"Scenario with {asset_name} price changing by {int(price_change_pct*100)}%",
            "adjustments": {
                "specific_assets": {
                    asset_key: {
                        "return_adjustment": price_change_pct,
                        "volatility_adjustment": volatility_adjustment
                    }
                }
            }
        }
    
    def _apply_scenario_adjustments(self, 
                                  portfolio_config: Dict[str, Any],
                                  scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Apply scenario adjustments to portfolio configuration"""
        adjustments = scenario.get("adjustments", {})
        
        # Apply asset return adjustments
        if "asset_returns" in adjustments:
            asset_returns = adjustments["asset_returns"]
            
            for asset in portfolio_config.get("assets", []):
                asset_type = asset.get("type", "")
                
                # Check if this asset type has an adjustment
                if asset_type in asset_returns:
                    # Apply return adjustment
                    adjustment = asset_returns[asset_type]
                    
                    # Adjustment can be absolute or relative
                    if isinstance(adjustment, float):
                        # For severe negative adjustments, ensure return becomes negative
                        if adjustment <= -0.3:
                            # Convert to absolute negative return
                            asset["return_override"] = adjustment
                        else:
                            # Apply as a relative adjustment
                            asset["return_multiplier"] = 1.0 + adjustment
        
        # Apply specific asset adjustments
        if "specific_assets" in adjustments:
            specific_adjustments = adjustments["specific_assets"]
            
            for asset in portfolio_config.get("assets", []):
                asset_type = asset.get("type", "")
                asset_name = asset.get("name", "")
                asset_key = f"{asset_type}_{asset_name}"
                
                if asset_key in specific_adjustments:
                    asset_adj = specific_adjustments[asset_key]
                    
                    # Apply specific return adjustment
                    if "return_adjustment" in asset_adj:
                        adjustment = asset_adj["return_adjustment"]
                        if adjustment <= -0.3:
                            asset["return_override"] = adjustment
                        else:
                            asset["return_multiplier"] = 1.0 + adjustment
                    
                    # Apply specific volatility adjustment
                    if "volatility_adjustment" in asset_adj:
                        asset["volatility_multiplier"] = asset_adj["volatility_adjustment"]
        
        # Apply volatility adjustments
        if "asset_volatility" in adjustments:
            volatility_adjs = adjustments["asset_volatility"]
            
            # Check for global adjustment
            if "global" in volatility_adjs:
                global_adj = volatility_adjs["global"]
                for asset in portfolio_config.get("assets", []):
                    asset["volatility_multiplier"] = global_adj
            
            # Apply specific asset type adjustments
            for asset in portfolio_config.get("assets", []):
                asset_type = asset.get("type", "")
                if asset_type in volatility_adjs:
                    asset["volatility_multiplier"] = volatility_adjs[asset_type]
        
        # Add correlation adjustments to portfolio config
        if "correlation_adjustments" in adjustments:
            portfolio_config["correlation_adjustments"] = adjustments["correlation_adjustments"]
        
        return portfolio_config
    
    def _compare_results(self,
                       baseline_results: Dict[str, Any],
                       scenario_results: Dict[str, Any],
                       scenario_name: str) -> Dict[str, Any]:
        """Compare baseline and scenario results"""
        baseline_stats = baseline_results["statistics"]
        scenario_stats = scenario_results["statistics"]
        
        # Calculate percentage changes
        mean_change_pct = (scenario_stats["mean"] - baseline_stats["mean"]) / baseline_stats["mean"]
        median_change_pct = (scenario_stats["median"] - baseline_stats["median"]) / baseline_stats["median"]
        
        # VaR change
        var_change_pct = (scenario_stats["var_95"] - baseline_stats["var_95"]) / baseline_stats["var_95"]
        
        # Drawdown change
        drawdown_change_pct = (scenario_stats["avg_max_drawdown"] - baseline_stats["avg_max_drawdown"]) / baseline_stats["avg_max_drawdown"]
        
        # Calculate risk metrics
        return {
            "scenario_name": scenario_name,
            "mean_change": {
                "absolute": scenario_stats["mean"] - baseline_stats["mean"],
                "percentage": mean_change_pct
            },
            "median_change": {
                "absolute": scenario_stats["median"] - baseline_stats["median"],
                "percentage": median_change_pct
            },
            "risk_measures": {
                "var_change": {
                    "absolute": scenario_stats["var_95"] - baseline_stats["var_95"],
                    "percentage": var_change_pct
                },
                "cvar_change": {
                    "absolute": scenario_stats["cvar_95"] - baseline_stats["cvar_95"],
                    "percentage": (scenario_stats["cvar_95"] - baseline_stats["cvar_95"]) / baseline_stats["cvar_95"]
                },
                "drawdown_change": {
                    "absolute": scenario_stats["avg_max_drawdown"] - baseline_stats["avg_max_drawdown"],
                    "percentage": drawdown_change_pct
                }
            },
            "probability_of_loss": self._calculate_probability_of_loss(
                baseline_results["simulation_paths"],
                scenario_results["simulation_paths"]
            ),
            "recovery_analysis": self._calculate_recovery_metrics(
                baseline_results["simulation_paths"],
                scenario_results["simulation_paths"]
            )
        }
    
    def _calculate_probability_of_loss(self,
                                     baseline_paths: np.ndarray,
                                     scenario_paths: np.ndarray) -> Dict[str, float]:
        """Calculate probability of various loss thresholds"""
        baseline_final = baseline_paths[:, -1]
        scenario_final = scenario_paths[:, -1]
        
        initial_value = baseline_paths[0, 0]
        
        # Calculate probability of different loss thresholds
        prob_any_loss_baseline = np.mean(baseline_final < initial_value)
        prob_any_loss_scenario = np.mean(scenario_final < initial_value)
        
        prob_10pct_loss_baseline = np.mean(baseline_final < initial_value * 0.9)
        prob_10pct_loss_scenario = np.mean(scenario_final < initial_value * 0.9)
        
        prob_25pct_loss_baseline = np.mean(baseline_final < initial_value * 0.75)
        prob_25pct_loss_scenario = np.mean(scenario_final < initial_value * 0.75)
        
        prob_50pct_loss_baseline = np.mean(baseline_final < initial_value * 0.5)
        prob_50pct_loss_scenario = np.mean(scenario_final < initial_value * 0.5)
        
        return {
            "probability_any_loss": {
                "baseline": prob_any_loss_baseline,
                "scenario": prob_any_loss_scenario,
                "difference": prob_any_loss_scenario - prob_any_loss_baseline
            },
            "probability_10pct_loss": {
                "baseline": prob_10pct_loss_baseline,
                "scenario": prob_10pct_loss_scenario,
                "difference": prob_10pct_loss_scenario - prob_10pct_loss_baseline
            },
            "probability_25pct_loss": {
                "baseline": prob_25pct_loss_baseline,
                "scenario": prob_25pct_loss_scenario,
                "difference": prob_25pct_loss_scenario - prob_25pct_loss_baseline
            },
            "probability_50pct_loss": {
                "baseline": prob_50pct_loss_baseline,
                "scenario": prob_50pct_loss_scenario,
                "difference": prob_50pct_loss_scenario - prob_50pct_loss_baseline
            }
        }
    
    def _calculate_recovery_metrics(self,
                                  baseline_paths: np.ndarray,
                                  scenario_paths: np.ndarray) -> Dict[str, Any]:
        """Calculate recovery-related metrics"""
        # Assuming monthly time steps
        num_sims, num_periods = scenario_paths.shape
        initial_value = baseline_paths[0, 0]
        
        # Calculate recovery time for each simulation path
        recovery_times = []
        for i in range(num_sims):
            path = scenario_paths[i, :]
            
            # Find minimum value in the path
            min_value = np.min(path)
            min_idx = np.argmin(path)
            
            # If the path recovers after the minimum, record time to recovery
            if min_idx < num_periods - 1:
                recovery_idx = min_idx
                while recovery_idx < num_periods - 1 and path[recovery_idx] < initial_value:
                    recovery_idx += 1
                
                if path[recovery_idx] >= initial_value:
                    # Found recovery point (in months)
                    recovery_time = recovery_idx - min_idx
                    recovery_times.append(recovery_time)
        
        # Calculate average recovery time (in months)
        if recovery_times:
            avg_recovery_time = np.mean(recovery_times)
            median_recovery_time = np.median(recovery_times)
        else:
            avg_recovery_time = None
            median_recovery_time = None
        
        # Calculate percentage of paths that recover
        paths_that_recover = len(recovery_times) / num_sims
        
        return {
            "paths_that_recover_pct": paths_that_recover,
            "avg_recovery_time_months": avg_recovery_time,
            "median_recovery_time_months": median_recovery_time
        }