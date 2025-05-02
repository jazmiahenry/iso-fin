import numpy as np
from typing import Dict, Any, List, Optional, Tuple

class BayesianScenarioRanker:
    def __init__(self):
        """Initialize the Bayesian Scenario Ranking Engine"""
        self.scenarios = {}
        self.scenario_priors = {}
        self.portfolio_impact = {}
        
    def add_scenario(self, 
                    scenario_id: str, 
                    name: str, 
                    description: str, 
                    prior_probability: float = 0.1,
                    category: Optional[str] = None) -> None:
        """
        Add a new scenario to the ranking engine
        
        Args:
            scenario_id: Unique identifier for the scenario
            name: Human-readable name
            description: Detailed description of the scenario
            prior_probability: Initial probability assigned to the scenario
            category: Optional category for grouping related scenarios
        """
        self.scenarios[scenario_id] = {
            "name": name,
            "description": description,
            "category": category,
            "features": {}
        }
        
        self.scenario_priors[scenario_id] = prior_probability
    
    def add_bulk_scenarios(self, scenarios: List[Dict[str, Any]]) -> None:
        """
        Add multiple scenarios at once
        
        Args:
            scenarios: List of scenario dictionaries
        """
        for scenario in scenarios:
            self.add_scenario(
                scenario_id=scenario.get("id"),
                name=scenario.get("name"),
                description=scenario.get("description"),
                prior_probability=scenario.get("prior_probability", 0.1),
                category=scenario.get("category")
            )
    
    def set_scenario_impact(self, 
                          scenario_id: str,
                          impact_metrics: Dict[str, float]) -> None:
        """
        Set the impact of a scenario on portfolio metrics
        
        Args:
            scenario_id: ID of the scenario
            impact_metrics: Dictionary of impact metrics
        """
        if scenario_id not in self.scenarios:
            raise ValueError(f"Scenario {scenario_id} not found")
            
        self.portfolio_impact[scenario_id] = impact_metrics
    
    def add_scenario_feature(self,
                           scenario_id: str,
                           feature_name: str,
                           feature_values: Dict[str, float]) -> None:
        """
        Add a feature to a scenario for Naive Bayes classification
        
        Args:
            scenario_id: ID of the scenario
            feature_name: Name of the feature
            feature_values: Dictionary mapping feature values to probabilities
        """
        if scenario_id not in self.scenarios:
            raise ValueError(f"Scenario {scenario_id} not found")
            
        self.scenarios[scenario_id]["features"][feature_name] = feature_values
    
    def update_scenario_probability(self,
                                  scenario_id: str,
                                  evidence: Dict[str, str]) -> float:
        """
        Update scenario probability based on new evidence using Naive Bayes
        
        Args:
            scenario_id: ID of the scenario
            evidence: Dictionary mapping feature names to observed values
            
        Returns:
            Updated probability for the scenario
        """
        if scenario_id not in self.scenarios:
            raise ValueError(f"Scenario {scenario_id} not found")
            
        scenario = self.scenarios[scenario_id]
        prior = self.scenario_priors[scenario_id]
        
        # Simple Naive Bayes update
        # P(scenario|evidence) ∝ P(evidence|scenario) × P(scenario)
        likelihood = 1.0
        
        for feature_name, observed_value in evidence.items():
            if feature_name in scenario["features"]:
                feature_values = scenario["features"][feature_name]
                
                # If this specific value has a probability, use it
                if observed_value in feature_values:
                    likelihood *= feature_values[observed_value]
                else:
                    # Use a small default probability for unobserved values
                    likelihood *= 0.01
        
        # Note: This doesn't normalize across all scenarios yet
        unnormalized_posterior = prior * likelihood
        
        return unnormalized_posterior
    
    def rank_scenarios(self, 
                      evidence: Dict[str, str] = None,
                      portfolio_composition: Dict[str, float] = None) -> List[Dict[str, Any]]:
        """
        Rank scenarios based on probability and portfolio impact
        
        Args:
            evidence: Optional dictionary of observed evidence
            portfolio_composition: Optional portfolio asset allocation
            
        Returns:
            List of scenarios ranked by relevance score
        """
        results = []
        
        # Calculate unnormalized posteriors
        posteriors = {}
        
        for scenario_id in self.scenarios:
            if evidence:
                # Update with evidence
                posteriors[scenario_id] = self.update_scenario_probability(
                    scenario_id, evidence
                )
            else:
                # Use priors
                posteriors[scenario_id] = self.scenario_priors[scenario_id]
        
        # Normalize posteriors
        total_probability = sum(posteriors.values())
        if total_probability > 0:
            for scenario_id in posteriors:
                posteriors[scenario_id] /= total_probability
        
        # Calculate portfolio-specific impact if portfolio composition is provided
        impact_scores = {}
        
        if portfolio_composition and self.portfolio_impact:
            for scenario_id in self.scenarios:
                if scenario_id in self.portfolio_impact:
                    impact = self.portfolio_impact[scenario_id]
                    
                    # Calculate weighted impact based on portfolio composition
                    weighted_impact = 0
                    
                    for asset_type, weight in portfolio_composition.items():
                        if asset_type in impact:
                            weighted_impact += weight * impact[asset_type]
                    
                    impact_scores[scenario_id] = abs(weighted_impact)
                else:
                    # Default moderate impact
                    impact_scores[scenario_id] = 0.5
        else:
            # Without portfolio data, all scenarios have equal impact
            for scenario_id in self.scenarios:
                impact_scores[scenario_id] = 0.5
        
        # Combine probability and impact for final ranking
        for scenario_id, posterior in posteriors.items():
            impact = impact_scores.get(scenario_id, 0.5)
            
            # Relevance score = probability * impact
            relevance = posterior * impact
            
            results.append({
                "scenario_id": scenario_id,
                "name": self.scenarios[scenario_id]["name"],
                "description": self.scenarios[scenario_id]["description"],
                "category": self.scenarios[scenario_id].get("category"),
                "probability": posterior,
                "impact": impact,
                "relevance": relevance
            })
        
        # Sort by relevance (descending)
        results.sort(key=lambda x: x["relevance"], reverse=True)
        
        return results
    
    def initialize_common_scenarios(self) -> None:
        """Initialize a set of common financial scenarios"""
        common_scenarios = [
            {
                "id": "crypto_winter",
                "name": "Crypto Winter",
                "description": "Extended bear market in crypto with 50%+ drawdowns",
                "prior_probability": 0.15,
                "category": "crypto"
            },
            {
                "id": "nft_crash",
                "name": "NFT Market Crash",
                "description": "Sharp decline in NFT floor prices and trading volume",
                "prior_probability": 0.2,
                "category": "nft"
            },
            {
                "id": "eth_scalability",
                "name": "ETH Scalability Breakthrough",
                "description": "Major Ethereum scalability solution drives adoption",
                "prior_probability": 0.1,
                "category": "crypto"
            },
            {
                "id": "defi_regulation",
                "name": "DeFi Regulatory Crackdown",
                "description": "Major regulatory action against DeFi protocols",
                "prior_probability": 0.25,
                "category": "regulatory"
            },
            {
                "id": "inflation_surge",
                "name": "Inflation Surge",
                "description": "Unexpected spike in inflation above central bank targets",
                "prior_probability": 0.3,
                "category": "macro"
            },
            {
                "id": "recession",
                "name": "Economic Recession",
                "description": "Global or regional economic recession",
                "prior_probability": 0.2,
                "category": "macro"
            },
            {
                "id": "real_estate_correction",
                "name": "Real Estate Market Correction",
                "description": "20%+ decline in real estate valuations",
                "prior_probability": 0.15,
                "category": "real_estate"
            },
            {
                "id": "metaverse_adoption",
                "name": "Metaverse Mainstream Adoption",
                "description": "Accelerated adoption of metaverse platforms and assets",
                "prior_probability": 0.1,
                "category": "tech"
            },
            {
                "id": "cbdc_launch",
                "name": "Major CBDC Launch",
                "description": "Launch of central bank digital currency by major economy",
                "prior_probability": 0.4,
                "category": "regulatory"
            },
            {
                "id": "crypto_etf_approval",
                "name": "Spot Crypto ETF Approval",
                "description": "Regulatory approval of spot cryptocurrency ETFs",
                "prior_probability": 0.3,
                "category": "regulatory"
            }
        ]
        
        self.add_bulk_scenarios(common_scenarios)
        
        # Add some example features for Naive Bayes
        
        # Crypto winter features
        self.add_scenario_feature(
            "crypto_winter",
            "btc_trend",
            {"downtrend": 0.8, "sideways": 0.5, "uptrend": 0.1}
        )
        self.add_scenario_feature(
            "crypto_winter",
            "exchange_volume",
            {"decreasing": 0.7, "stable": 0.4, "increasing": 0.1}
        )
        
        # NFT crash features
        self.add_scenario_feature(
            "nft_crash",
            "nft_volume",
            {"decreasing": 0.8, "stable": 0.3, "increasing": 0.1}
        )
        self.add_scenario_feature(
            "nft_crash",
            "floor_prices",
            {"decreasing": 0.8, "stable": 0.2, "increasing": 0.05}
        )
        
        # Regulatory features for multiple scenarios
        for scenario_id in ["defi_regulation", "cbdc_launch", "crypto_etf_approval"]:
            self.add_scenario_feature(
                scenario_id,
                "regulatory_news",
                {"negative": 0.7, "neutral": 0.3, "positive": 0.1}
            )
            self.add_scenario_feature(
                scenario_id,
                "government_statements",
                {"restrictive": 0.8, "neutral": 0.4, "supportive": 0.1}
            )
        
        # Macro features for inflation and recession
        for scenario_id in ["inflation_surge", "recession"]:
            self.add_scenario_feature(
                scenario_id,
                "central_bank_policy",
                {"hawkish": 0.7, "neutral": 0.4, "dovish": 0.2}
            )
            self.add_scenario_feature(
                scenario_id,
                "economic_indicators",
                {"negative": 0.7, "mixed": 0.4, "positive": 0.1}
            )
        
        # Add example portfolio impacts
        self.set_scenario_impact(
            "crypto_winter",
            {
                "crypto": -0.6,
                "nft": -0.7,
                "real_estate": -0.1,
                "stocks": -0.2,
                "bonds": 0.1
            }
        )
        
        self.set_scenario_impact(
            "inflation_surge",
            {
                "crypto": 0.2,
                "real_estate": 0.3,
                "stocks": -0.1,
                "bonds": -0.4
            }
        )
    
    def update_all_scenarios(self, evidence: Dict[str, str]) -> Dict[str, float]:
        """
        Update probabilities for all scenarios based on new evidence
        
        Args:
            evidence: Dictionary of evidence
            
        Returns:
            Dictionary mapping scenario IDs to updated probabilities
        """
        # Calculate unnormalized posteriors
        posteriors = {}
        
        for scenario_id in self.scenarios:
            posteriors[scenario_id] = self.update_scenario_probability(
                scenario_id, evidence
            )
        
        # Normalize posteriors
        total_probability = sum(posteriors.values())
        if total_probability > 0:
            for scenario_id in posteriors:
                posteriors[scenario_id] /= total_probability
        
        # Update scenario priors with new posteriors
        self.scenario_priors = posteriors
        
        return posteriors
    
    def get_scenario_details(self, scenario_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific scenario
        
        Args:
            scenario_id: ID of the scenario
            
        Returns:
            Dictionary with scenario details
        """
        if scenario_id not in self.scenarios:
            raise ValueError(f"Scenario {scenario_id} not found")
            
        scenario = self.scenarios[scenario_id]
        probability = self.scenario_priors[scenario_id]
        impact = self.portfolio_impact.get(scenario_id, {})
        
        return {
            "id": scenario_id,
            "name": scenario["name"],
            "description": scenario["description"],
            "category": scenario.get("category"),
            "probability": probability,
            "impact": impact,
            "features": scenario.get("features", {})
        }