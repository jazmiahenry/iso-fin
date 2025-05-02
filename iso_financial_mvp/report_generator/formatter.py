import json
from typing import Dict, Any, List, Optional
import os
from jinja2 import Environment, FileSystemLoader
import datetime

class ReportFormatter:
    def __init__(self):
        """Initialize the report formatter"""
        # Setup Jinja2 environment
        template_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates")
        self.jinja_env = Environment(loader=FileSystemLoader(template_dir))
        
    def format_report(self, 
                     simulation_results: Dict[str, Any],
                     query_data: Dict[str, Any],
                     user_profile: str = "novice") -> Dict[str, Any]:
        """
        Format simulation results into a user-friendly report
        
        Args:
            simulation_results: Results from simulation engines
            query_data: Original query data
            user_profile: User expertise level (novice, intermediate, pro)
            
        Returns:
            Formatted report
        """
        # In a production environment, we'd use an LLM to generate natural language
        # explanations. For the MVP, we'll use templates.
        
        # Load template based on query type
        query_type = query_data.get("query_type", "portfolio_analysis")
        
        # Extract key information
        report = {
            "query": query_data.get("original_query", ""),
            "timestamp": datetime.datetime.now().isoformat(),
            "user_profile": user_profile,
            "summary": self._generate_summary(simulation_results, query_data, user_profile),
            "detailed_results": self._format_detailed_results(simulation_results, query_type),
            "recommendations": self._generate_recommendations(simulation_results, query_data, user_profile),
            "visualizations": self._generate_visualization_configs(simulation_results, query_type),
            "raw_data": {
                "query": query_data,
                "results": self._clean_numpy_arrays(simulation_results)
            }
        }
        
        return report
    
    def _generate_summary(self, 
                        simulation_results: Dict[str, Any],
                        query_data: Dict[str, Any],
                        user_profile: str) -> str:
        """Generate a natural language summary of the results"""
        # Load the response template
        try:
            template = self.jinja_env.get_template("response_prompt.j2")
            prompt = template.render(
                original_query=query_data.get("original_query", ""),
                results=simulation_results,
                user_profile=user_profile
            )
            
            # In production, this would send the prompt to an LLM
            # For now, return a template-based summary
            return self._get_template_summary(simulation_results, query_data)
            
        except Exception as e:
            # Fallback if template loading fails
            return self._get_template_summary(simulation_results, query_data)
    
    def _get_template_summary(self, 
                            simulation_results: Dict[str, Any],
                            query_data: Dict[str, Any]) -> str:
        """Generate a summary based on templates when LLM is not available"""
        query_type = query_data.get("query_type", "portfolio_analysis")
        
        summary_templates = {
            "portfolio_analysis": "Based on the analysis of your portfolio over {timeframe}, "
                                 "the projected value is ${projected_value:,.2f} with a "
                                 "{confidence_level}% confidence interval of "
                                 "${confidence_lower:,.2f} to ${confidence_upper:,.2f}.",
            
            "risk_assessment": "Your portfolio has a {risk_level} risk profile. "
                             "There is a {var_percent}% chance of losing ${var_amount:,.2f} or more "
                             "over the specified time period. The maximum drawdown could reach {max_drawdown:.1%}.",
            
            "scenario_analysis": "In the {scenario_name} scenario, your portfolio value would change "
                               "by {value_change:.1%}. The probability of a loss increases from {baseline_prob:.1%} "
                               "to {scenario_prob:.1%}.",
            
            "retirement_planning": "Based on your current portfolio and assumptions, "
                                 "you have a {retirement_probability:.1%} chance of reaching your retirement goal "
                                 "in {retirement_years} years. Your projected portfolio value at retirement "
                                 "is ${retirement_value:,.2f}."
        }
        
        # Extract data based on query type
        if query_type == "portfolio_analysis":
            if "monte_carlo" in simulation_results:
                mc_results = simulation_results["monte_carlo"]
                stats = mc_results.get("statistics", {})
                
                timeframe_value = query_data.get("timeframe", {}).get("value", 10)
                timeframe_unit = query_data.get("timeframe", {}).get("unit", "years")
                timeframe = f"{timeframe_value} {timeframe_unit}"
                
                confidence_intervals = stats.get("confidence_intervals", {})
                
                return summary_templates[query_type].format(
                    timeframe=timeframe,
                    projected_value=stats.get("mean", 0),
                    confidence_level=90,
                    confidence_lower=confidence_intervals.get("p5", 0),
                    confidence_upper=confidence_intervals.get("p95", 0)
                )
        
        elif query_type == "risk_assessment":
            if "cvar" in simulation_results:
                cvar_results = simulation_results["cvar"]
                
                risk_profile = query_data.get("risk_profile", "moderate")
                risk_level_map = {"conservative": "low", "moderate": "medium", "aggressive": "high"}
                risk_level = risk_level_map.get(risk_profile, "medium")
                
                return summary_templates[query_type].format(
                    risk_level=risk_level,
                    var_percent=5,
                    var_amount=cvar_results.get("var_amount", 0),
                    max_drawdown=cvar_results.get("maximum_drawdown", {}).get("average", 0.15)
                )
        
        elif query_type == "scenario_analysis":
            if "scenario" in simulation_results:
                scenario_results = simulation_results["scenario"]
                comparison = scenario_results.get("comparison", {})
                scenario_details = scenario_results.get("scenario_details", {})
                
                baseline_prob = scenario_results.get("baseline", {}).get("probability_of_loss", 0.2)
                scenario_prob = scenario_results.get("scenario", {}).get("probability_of_loss", 0.3)
                
                mean_change = comparison.get("mean_change", {}).get("percentage", 0)
                
                return summary_templates[query_type].format(
                    scenario_name=scenario_details.get("name", "analyzed"),
                    value_change=mean_change,
                    baseline_prob=baseline_prob,
                    scenario_prob=scenario_prob
                )
        
        elif query_type == "retirement_planning":
            if "monte_carlo" in simulation_results:
                mc_results = simulation_results["monte_carlo"]
                stats = mc_results.get("statistics", {})
                
                timeframe_value = query_data.get("timeframe", {}).get("value", 20)
                
                # Calculate probability of reaching target
                confidence_intervals = stats.get("confidence_intervals", {})
                target_value = query_data.get("retirement_target", 1000000)
                
                # Approximate probability of reaching target
                mean = stats.get("mean", 0)
                std = stats.get("std", 0)
                if std > 0:
                    import math
                    z_score = (target_value - mean) / std
                    # Approximating normal CDF
                    retirement_probability = 0.5 * (1 - math.erf(z_score / math.sqrt(2)))
                else:
                    retirement_probability = 0.5
                
                return summary_templates[query_type].format(
                    retirement_probability=1-retirement_probability,
                    retirement_years=timeframe_value,
                    retirement_value=mean
                )
        
        # Default summary if specific data not found
        return "Analysis complete. Please see the detailed results below."
    
    def _format_detailed_results(self, 
                               simulation_results: Dict[str, Any],
                               query_type: str) -> Dict[str, Any]:
        """Format detailed results based on query type"""
        detailed_results = {}
        
        if query_type == "portfolio_analysis" or query_type == "retirement_planning":
            if "monte_carlo" in simulation_results:
                mc_results = simulation_results["monte_carlo"]
                stats = mc_results.get("statistics", {})
                
                detailed_results["portfolio_statistics"] = {
                    "mean_final_value": stats.get("mean", 0),
                    "median_final_value": stats.get("median", 0),
                    "standard_deviation": stats.get("std", 0),
                    "confidence_intervals": stats.get("confidence_intervals", {}),
                }
                
                detailed_results["risk_metrics"] = {
                    "value_at_risk": stats.get("var_95", 0),
                    "conditional_value_at_risk": stats.get("cvar_95", 0),
                    "maximum_drawdown": stats.get("max_drawdown", 0),
                    "average_maximum_drawdown": stats.get("avg_max_drawdown", 0),
                }
                
        elif query_type == "risk_assessment":
            if "cvar" in simulation_results:
                cvar_results = simulation_results["cvar"]
                
                detailed_results["value_at_risk"] = {
                    "var_95": cvar_results.get("var_amount", 0),
                    "cvar_95": cvar_results.get("cvar_amount", 0),
                    "var_90": cvar_results.get("comparison", {}).get("var_90", 0),
                    "var_99": cvar_results.get("comparison", {}).get("var_99", 0),
                }
                
                detailed_results["drawdown_risk"] = cvar_results.get("drawdown_risk", {})
                
                if "stress_scenarios" in cvar_results:
                    detailed_results["stress_test"] = cvar_results["stress_scenarios"]
            
            if "risk_parity" in simulation_results:
                rp_results = simulation_results["risk_parity"]
                
                detailed_results["risk_contribution"] = {
                    "portfolio_volatility": rp_results.get("portfolio_volatility", 0),
                    "diversification_ratio": rp_results.get("diversification_ratio", 0),
                    "asset_details": rp_results.get("asset_details", []),
                }
                
                if "trades" in rp_results:
                    detailed_results["recommended_trades"] = rp_results["trades"]
        
        elif query_type == "scenario_analysis":
            if "scenario" in simulation_results:
                scenario_results = simulation_results["scenario"]
                
                detailed_results["scenario_details"] = scenario_results.get("scenario_details", {})
                detailed_results["comparison"] = scenario_results.get("comparison", {})
                
                # Add recovery analysis if available
                if "recovery_analysis" in scenario_results.get("comparison", {}):
                    detailed_results["recovery_analysis"] = scenario_results["comparison"]["recovery_analysis"]
            
            if "bayesian" in simulation_results:
                bayesian_results = simulation_results["bayesian"]
                
                detailed_results["scenario_rankings"] = bayesian_results.get("ranked_scenarios", [])
                detailed_results["scenario_probabilities"] = bayesian_results.get("updated_probabilities", {})
        
        return detailed_results
    
    def _generate_recommendations(self,
                                simulation_results: Dict[str, Any],
                                query_data: Dict[str, Any],
                                user_profile: str) -> List[Dict[str, str]]:
        """Generate recommendations based on simulation results"""
        recommendations = []
        query_type = query_data.get("query_type", "portfolio_analysis")
        risk_profile = query_data.get("risk_profile", "moderate")
        
        # Portfolio diversity recommendation
        if "monte_carlo" in simulation_results:
            mc_results = simulation_results["monte_carlo"]
            
            # Check if portfolio is heavily concentrated
            assets = query_data.get("assets", [])
            asset_types = [asset.get("type", "") for asset in assets]
            
            if len(set(asset_types)) < 3:
                recommendations.append({
                    "type": "diversification",
                    "title": "Increase Portfolio Diversification",
                    "description": "Your portfolio appears concentrated in a small number of asset types. "
                                 "Consider adding more diverse assets to reduce overall risk."
                })
            
            # Check drawdown risk
            stats = mc_results.get("statistics", {})
            max_drawdown = stats.get("max_drawdown", 0)
            
            if max_drawdown > 0.3 and risk_profile != "aggressive":
                recommendations.append({
                    "type": "risk_mitigation",
                    "title": "Consider Reducing Volatility",
                    "description": f"Your portfolio shows a potential maximum drawdown of {max_drawdown:.1%}, "
                                 f"which may be higher than appropriate for your {risk_profile} risk profile."
                })
        
        # Risk parity recommendations
        if "risk_parity" in simulation_results:
            rp_results = simulation_results["risk_parity"]
            
            if "trades" in rp_results and len(rp_results["trades"]) > 0:
                recommendations.append({
                    "type": "rebalancing",
                    "title": "Rebalance for Better Risk Distribution",
                    "description": "Rebalancing your portfolio according to risk parity principles "
                                 "could improve your risk-adjusted returns."
                })
        
        # Scenario-specific recommendations
        if "scenario" in simulation_results:
            scenario_results = simulation_results["scenario"]
            scenario_details = scenario_results.get("scenario_details", {})
            comparison = scenario_results.get("comparison", {})
            
            # If significant negative impact, recommend hedging
            mean_change = comparison.get("mean_change", {}).get("percentage", 0)
            
            if mean_change < -0.15:
                scenario_name = scenario_details.get("name", "analyzed")
                recommendations.append({
                    "type": "hedging",
                    "title": f"Hedge Against {scenario_name}",
                    "description": f"Your portfolio could lose {abs(mean_change):.1%} in the {scenario_name} scenario. "
                                 f"Consider hedging strategies to protect against this risk."
                })
        
        # Add general recommendations based on user profile
        if user_profile == "novice":
            recommendations.append({
                "type": "education",
                "title": "Continue Learning About Asset Classes",
                "description": "As you're new to investing, spend time understanding the different "
                             "asset classes in your portfolio and their risk characteristics."
            })
        elif user_profile == "pro":
            recommendations.append({
                "type": "advanced_strategy",
                "title": "Consider Options Strategies",
                "description": "Given your advanced knowledge, explore using options strategies "
                             "to hedge specific risks or enhance returns in your portfolio."
            })
        
        return recommendations
    
    def _generate_visualization_configs(self,
                                      simulation_results: Dict[str, Any],
                                      query_type: str) -> Dict[str, Dict[str, Any]]:
        """Generate configurations for visualizations"""
        visualizations = {}
        
        # Monte Carlo simulation paths visualization
        if "monte_carlo" in simulation_results:
            mc_results = simulation_results["monte_carlo"]
            
            # We don't include the full paths data here, just the configuration
            visualizations["monte_carlo_paths"] = {
                "type": "line_chart",
                "title": "Portfolio Value Projection",
                "description": "Monte Carlo simulation of possible portfolio values over time",
                "data_ref": "monte_carlo.simulation_paths",
                "show_percentiles": [5, 50, 95]
            }
            
            # Final value distribution
            visualizations["value_distribution"] = {
                "type": "histogram",
                "title": "Final Portfolio Value Distribution",
                "description": "Distribution of possible portfolio values at the end of the simulation",
                "data_ref": "monte_carlo.simulation_paths[:, -1]",
                "bins": 30
            }
        
        # Risk parity visualization
        if "risk_parity" in simulation_results:
            rp_results = simulation_results["risk_parity"]
            
            visualizations["risk_contribution"] = {
                "type": "pie_chart",
                "title": "Risk Contribution by Asset",
                "description": "Percentage of total portfolio risk from each asset",
                "data_ref": "risk_parity.risk_contribution"
            }
            
            visualizations["optimal_allocation"] = {
                "type": "bar_chart",
                "title": "Optimal vs Current Allocation",
                "description": "Comparison of current and risk-parity optimized allocation",
                "data_ref": "risk_parity.asset_details",
                "compare_fields": ["current_weight", "weight"]
            }
        
        # Scenario analysis visualization
        if "scenario" in simulation_results:
            scenario_results = simulation_results["scenario"]
            
            visualizations["scenario_comparison"] = {
                "type": "bar_chart",
                "title": "Baseline vs Scenario Comparison",
                "description": "Key metrics compared between baseline and scenario",
                "data_ref": "scenario.comparison"
            }
            
            # Probability of loss chart
            if "probability_of_loss" in scenario_results.get("comparison", {}):
                visualizations["loss_probability"] = {
                    "type": "grouped_bar_chart",
                    "title": "Probability of Loss",
                    "description": "Probability of different loss thresholds in baseline vs scenario",
                    "data_ref": "scenario.comparison.probability_of_loss"
                }
        
        # CVaR visualization
        if "cvar" in simulation_results:
            cvar_results = simulation_results["cvar"]
            
            visualizations["tail_risk"] = {
                "type": "bar_chart",
                "title": "Tail Risk Measures",
                "description": "Value at Risk (VaR) and Conditional Value at Risk (CVaR) at different confidence levels",
                "data_ref": "cvar.comparison"
            }
            
            # Stress test visualization
            if "stress_scenarios" in cvar_results:
                visualizations["stress_test"] = {
                    "type": "line_chart",
                    "title": "Stress Test Results",
                    "description": "Impact of different stress scenarios on portfolio value",
                    "data_ref": "cvar.stress_scenarios"
                }
        
        # Bayesian scenario ranking visualization
        if "bayesian" in simulation_results:
            bayesian_results = simulation_results["bayesian"]
            
            visualizations["scenario_ranking"] = {
                "type": "horizontal_bar_chart",
                "title": "Scenario Risk Ranking",
                "description": "Scenarios ranked by relevance to your portfolio",
                "data_ref": "bayesian.ranked_scenarios",
                "x_field": "relevance",
                "y_field": "name",
                "color_field": "category"
            }
        
        return visualizations
    
    def _clean_numpy_arrays(self, data: Any) -> Any:
        """Convert numpy arrays to lists for JSON serialization"""
        import numpy as np
        
        if isinstance(data, dict):
            return {k: self._clean_numpy_arrays(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._clean_numpy_arrays(item) for item in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, np.integer):
            return int(data)
        elif isinstance(data, np.floating):
            return float(data)
        elif isinstance(data, np.bool_):
            return bool(data)
        else:
            return data
            
    def export_report_to_json(self, report: Dict[str, Any], filename: str) -> str:
        """
        Export the report to a JSON file
        
        Args:
            report: Formatted report
            filename: Output filename
            
        Returns:
            Path to the output file
        """
        # Ensure the filename has .json extension
        if not filename.endswith('.json'):
            filename += '.json'
        
        # Remove numpy arrays for JSON serialization
        clean_report = self._clean_numpy_arrays(report)
        
        # Write to file
        with open(filename, 'w') as f:
            json.dump(clean_report, f, indent=2)
        
        return filename