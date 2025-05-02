import json
import os
from typing import Dict, Any, List, Optional
from jinja2 import Environment, FileSystemLoader

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Import Anthropic for Claude API
try:
    import anthropic
except ImportError:
    print("WARNING: Anthropic package not installed. Install with pip install anthropic")
    anthropic = None

class QueryParser:
    def __init__(self, model_name="claude-3-sonnet-20240229"):
        # Ensure valid model name for Claude API
        if model_name == "claude-3-7-sonnet-20250219":
            # Newer model - standardize to a known model that exists in the API
            self.model_name = "claude-3-sonnet-20240229"
        else:
            self.model_name = model_name
            
        self.supported_query_types = [
            "portfolio_analysis", 
            "risk_assessment", 
            "scenario_analysis", 
            "retirement_planning"
        ]
        
        # Setup Jinja2 environment
        template_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "templates")
        self.jinja_env = Environment(loader=FileSystemLoader(template_dir))
        
        # Get Anthropic API key for Claude
        self.claude_api_key = os.getenv("ANTHROPIC_API_KEY")
        
        # Add Claude/OpenAI flags for compatibility with existing code
        self.use_claude = True
        self.use_openai = False
    
    def parse_query(self, natural_language_query: str) -> Dict[str, Any]:
        """
        Parse natural language query into structured format for routing to appropriate models
        
        Args:
            natural_language_query: User's question in natural language
            
        Returns:
            Structured dictionary with query intent and parameters
        """
        # Use rule-based parsing logic for all queries
        query_type = self._determine_query_type(natural_language_query)
        timeframe = self._extract_timeframe(natural_language_query)
        assets = self._extract_assets(natural_language_query)
        events = self._extract_events(natural_language_query)
        risk_profile = self._extract_risk_profile(natural_language_query)
        
        return {
            "query_type": query_type,
            "timeframe": timeframe,
            "assets": assets,
            "events": events,
            "risk_profile": risk_profile,
            "original_query": natural_language_query
        }
    
    def _determine_query_type(self, query: str) -> str:
        """Determine the type of financial query based on keywords"""
        query = query.lower()
        
        if any(word in query for word in ["risk", "risky", "safe", "volatility"]):
            return "risk_assessment"
        elif any(word in query for word in ["retire", "retirement", "future", "years"]):
            return "retirement_planning"
        elif any(word in query for word in ["if", "what if", "scenario", "crash", "drops", "rises"]):
            return "scenario_analysis"
        else:
            return "portfolio_analysis"
    
    def _extract_timeframe(self, query: str) -> Optional[Dict[str, Any]]:
        """Extract time-related parameters from the query"""
        query = query.lower()
        
        # Simple regex-like pattern matching - would be more sophisticated with actual LLM
        timeframe = None
        
        if "years" in query:
            for i in range(1, 51):  # Look for numbers 1-50
                if f"{i} years" in query or f"{i} year" in query:
                    timeframe = {"unit": "years", "value": i}
                    break
        
        if "months" in query:
            for i in range(1, 25):  # Look for numbers 1-24
                if f"{i} months" in query or f"{i} month" in query:
                    timeframe = {"unit": "months", "value": i}
                    break
        
        return timeframe or {"unit": "years", "value": 10}  # Default to 10 years
    
    def _extract_assets(self, query: str) -> List[Dict[str, Any]]:
        """Extract mentioned assets from the query"""
        query = query.lower()
        assets = []
        
        # In a real implementation, this would use entity recognition from the LLM
        if "eth" in query or "ethereum" in query:
            assets.append({"type": "crypto", "name": "ethereum"})
        
        if "nft" in query:
            assets.append({"type": "nft", "name": "general"})
        
        if "real estate" in query:
            assets.append({"type": "real_estate", "name": "general"})
        
        if "art" in query:
            assets.append({"type": "collectible", "name": "art"})
        
        # Return default portfolio if no assets detected
        if not assets:
            assets = [
                {"type": "crypto", "name": "general"},
                {"type": "traditional", "name": "stocks"}
            ]
        
        return assets
    
    def _extract_events(self, query: str) -> List[Dict[str, Any]]:
        """Extract event scenarios mentioned in the query"""
        query = query.lower()
        events = []
        
        # Check for specific events or scenarios
        if "crash" in query or "drop" in query:
            # Try to extract percentage if available
            import re
            percentages = re.findall(r'(\d+)%', query)
            
            if percentages:
                percentage = int(percentages[0])
                # Find which asset is mentioned
                if "eth" in query or "ethereum" in query:
                    events.append({
                        "type": "price_change",
                        "asset": "ethereum",
                        "change": -percentage / 100
                    })
                else:
                    events.append({
                        "type": "market_change",
                        "change": -percentage / 100
                    })
            else:
                events.append({"type": "market_crash", "severity": "medium"})
        
        if "regulation" in query or "regulatory" in query:
            events.append({"type": "regulatory_event", "severity": "high"})
        
        return events
    
    def _extract_risk_profile(self, query: str) -> str:
        """Determine risk profile from query language"""
        query = query.lower()
        
        if any(word in query for word in ["conservative", "safe", "low risk"]):
            return "conservative"
        elif any(word in query for word in ["moderate", "balanced"]):
            return "moderate"
        elif any(word in query for word in ["aggressive", "high risk", "risky"]):
            return "aggressive"
        
        # Default to moderate
        return "moderate"
    
    def generate_response(self, results: Dict[str, Any], user_profile: str = "novice") -> str:
        """
        Generate natural language response from model results using Claude API
        
        Args:
            results: Results from various simulation models
            user_profile: User expertise level (novice, intermediate, pro)
            
        Returns:
            Natural language explanation of results
        """
        # Format key statistics to make them easier to understand
        formatted_results = self._format_results_for_llm(results)
        
        # Extract key metrics
        initial_investment = formatted_results.get("simplified", {}).get("initial_investment", "$100,000")
        projected_value = formatted_results.get("simplified", {}).get("projected_value", "$0")
        annual_return = formatted_results.get("simplified", {}).get("annual_return", "0%")
        percentile_5 = formatted_results.get("simplified", {}).get("percentile_5", "$0")
        percentile_95 = formatted_results.get("simplified", {}).get("percentile_95", "$0")
        success_probability = formatted_results.get("simplified", {}).get("success_probability", "0%")
        
        # Extract risk metrics
        risk_metrics = formatted_results.get("simplified", {}).get("risk", {})
        var_95 = risk_metrics.get("var_95", "$0")
        cvar_95 = risk_metrics.get("cvar_95", "$0")
        max_drawdown = risk_metrics.get("max_drawdown", "0%")
        sharpe_ratio = risk_metrics.get("sharpe_ratio", "0.0")
        
        # Create query context
        query_data = results.get("query_data", {})
        query_type = query_data.get("query_type", "portfolio_analysis")
        timeframe = query_data.get("timeframe", {}).get("value", 10)
        risk_profile = query_data.get("risk_profile", "moderate")
        original_query = results.get("original_query", "Analyze my portfolio")
        
        # Extract portfolio information
        portfolio = query_data.get("portfolio", {})
        portfolio_str = ", ".join([f"{ticker}: {weight:.1%}" for ticker, weight in portfolio.items()])
        
        # Create a direct prompt for Claude
        system_prompt = "You are a financial advisor assistant that helps explain portfolio analysis results in clear, concise language appropriate for the user's knowledge level. Focus on key insights and actionable recommendations."
        
        user_prompt = f"""
User query: "{original_query}"

Portfolio Analysis Results:
- Initial Investment: {initial_investment}
- Projected Value (after {timeframe} years): {projected_value}
- Annual Return: {annual_return}
- 5th Percentile Outcome: {percentile_5}
- 95th Percentile Outcome: {percentile_95}
- Probability of Positive Return: {success_probability}

Risk Assessment:
- Value at Risk (95%): {var_95}
- Conditional Value at Risk (95%): {cvar_95}
- Maximum Drawdown: {max_drawdown}
- Sharpe Ratio: {sharpe_ratio}
- Risk Profile: {risk_profile}

Portfolio Composition:
{portfolio_str}

Additional Context:
- Query Type: {query_type}
- Time Horizon: {timeframe} years
- User Expertise: {user_profile}

Please provide a direct answer to the user's query based on these results. Include specific insights from the analysis and actionable recommendations. Use language appropriate for a {user_profile} user.
"""
        
        # Check if Claude API key and library are available
        if self.claude_api_key and anthropic:
            try:
                # Initialize the Anthropic client
                client = anthropic.Anthropic(api_key=self.claude_api_key)
                
                # Call Claude API - using proper method for the installed version
                response = client.messages.create(
                    model=self.model_name,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )
                
                # Extract the content from the response
                content = response.content[0].text
                print(f"Claude API response received: {content[:50]}...")
                return content
            except Exception as e:
                print(f"Error using Claude API: {str(e)}")
        
        # Fallback to rule-based response if Claude API is not available or fails
        if query_type == "portfolio_analysis":
            response = f"""Based on my analysis of your portfolio ({portfolio_str}) with an initial investment of {initial_investment}, the projected mean value after {timeframe} years is {projected_value}. This represents an annual return of {annual_return}.

In the best-case scenario (95th percentile), your portfolio could grow to {percentile_95}, while in a poor scenario (5th percentile), it might only reach {percentile_5}. The probability of achieving a positive return is {success_probability}.

Your portfolio's Value at Risk (95%) is {var_95}, and the maximum drawdown could reach {max_drawdown}. With a Sharpe ratio of {sharpe_ratio}, your portfolio's risk-adjusted return is reasonable.

I recommend regular portfolio rebalancing and considering adding more diversified assets to improve risk-adjusted returns.
"""
            
        elif query_type == "risk_assessment":
            response = f"""Your portfolio has a {risk_profile} risk profile with key risk metrics:
- Value at Risk (95%): {var_95}
- Conditional Value at Risk (95%): {cvar_95}
- Maximum Drawdown: {max_drawdown}
- Sharpe Ratio: {sharpe_ratio}

Based on these metrics, your portfolio appears to have a moderate risk level. To improve risk characteristics, consider adding more uncorrelated assets to reduce volatility while maintaining returns.
"""

        elif query_type == "scenario_analysis":
            response = f"""I've analyzed how your portfolio would perform under different economic scenarios.

In a market downturn, your portfolio could experience a maximum drawdown of {max_drawdown}. Your Value at Risk (95%) is {var_95}, indicating the potential loss in adverse conditions.

To better protect against adverse scenarios, consider adding defensive assets like bonds or gold to your portfolio if you don't already have them.
"""

        elif query_type == "retirement_planning":
            response = f"""Based on your portfolio ({portfolio_str}) and an initial investment of {initial_investment}, here's my analysis for your retirement planning over {timeframe} years:

Your expected portfolio value at the end of this period is {projected_value}, representing an annual return of {annual_return}. The probability of achieving a positive return is {success_probability}.

I recommend reviewing your retirement goals to ensure this projected outcome aligns with your needs. Consider increasing contributions if there appears to be a shortfall.
"""
        else:
            response = f"""Based on my analysis of your portfolio with an initial investment of {initial_investment}, 
the projected mean value after {timeframe} years is {projected_value}. 
Your portfolio has a {risk_profile} risk profile with an expected annual return of {annual_return}.

To improve your results, consider rebalancing regularly and adjusting your asset allocation based on your financial goals and risk tolerance.
"""
            
        return response
    
    def _format_results_for_llm(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Format results to be more LLM-friendly with key statistics and insights"""
        # Create a deep copy to avoid modifying the original
        formatted = {}
        
        # Copy only what we need, converting numpy arrays to lists
        for key, value in results.items():
            if key == "monte_carlo_results":
                mc_copy = {}
                for mc_key, mc_value in value.items():
                    if mc_key == "paths" and hasattr(mc_value, "tolist"):
                        # Skip paths array to avoid large data transfer
                        continue
                    elif mc_key == "percentiles" and isinstance(mc_value, dict):
                        # Handle percentiles dict with possible numpy values
                        mc_copy[mc_key] = {k: float(v) if hasattr(v, "item") else v 
                                         for k, v in mc_value.items()}
                    elif hasattr(mc_value, "tolist"):
                        # Convert numpy arrays to lists
                        mc_copy[mc_key] = mc_value.tolist()
                    elif hasattr(mc_value, "item"):
                        # Convert numpy scalars to Python scalars
                        mc_copy[mc_key] = mc_value.item()
                    else:
                        mc_copy[mc_key] = mc_value
                formatted[key] = mc_copy
            elif key == "risk_results" or key == "optimization_results" or key == "scenario_results":
                # Copy these dictionaries with numpy conversion
                formatted[key] = self._convert_numpy_to_python(value)
            else:
                # Just copy other keys directly
                formatted[key] = value
        
        # Extract and simplify monte carlo results
        mc_results = results.get("monte_carlo_results", {})
        if mc_results:
            initial_investment = mc_results.get("initial_investment", 100000)
            mean_final = mc_results.get("mean_final", 0)
            
            # Convert numpy types to Python types
            if hasattr(initial_investment, "item"):
                initial_investment = initial_investment.item()
            if hasattr(mean_final, "item"):
                mean_final = mean_final.item()
            
            # Calculate annualized return if possible
            if mean_final > 0 and initial_investment > 0:
                years = results.get("query_data", {}).get("timeframe", {}).get("value", 10)
                if years <= 0:
                    years = 10  # Default to 10 years if invalid
                ann_return = (mean_final / initial_investment) ** (1 / years) - 1
            else:
                ann_return = 0
                
            # Add simplified stats to formatted results
            formatted["simplified"] = {
                "initial_investment": f"${initial_investment:,.0f}",
                "projected_value": f"${mean_final:,.0f}",
                "annual_return": f"{ann_return:.2%}",
                "percentile_5": f"${self._get_percentile(mc_results, 5):,.0f}",
                "percentile_95": f"${self._get_percentile(mc_results, 95):,.0f}",
                "success_probability": f"{self._get_probability(mc_results):,.1%}"
            }
            
            # Add risk metrics
            risk_results = results.get("risk_results", {})
            formatted["simplified"]["risk"] = {
                "var_95": f"${self._get_value(risk_results, 'var_95', 0):,.0f}",
                "cvar_95": f"${self._get_value(risk_results, 'cvar_95', 0):,.0f}",
                "max_drawdown": f"{self._get_value(risk_results, 'max_drawdown', 0):.1%}",
                "sharpe_ratio": f"{self._get_value(risk_results, 'sharpe_ratio', 0):.2f}"
            }
        
        return formatted
    
    def _convert_numpy_to_python(self, obj):
        """Recursively convert numpy types to Python types in a dictionary or list"""
        if isinstance(obj, dict):
            return {k: self._convert_numpy_to_python(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_to_python(item) for item in obj]
        elif hasattr(obj, "tolist"):
            return obj.tolist()
        elif hasattr(obj, "item"):
            return obj.item()
        else:
            return obj
            
    def _get_percentile(self, mc_results, percentile):
        """Safely extract percentile values, converting numpy types if needed"""
        percentiles = mc_results.get('percentiles', {})
        if not percentiles:
            return 0
            
        value = percentiles.get(percentile, 0)
        if hasattr(value, "item"):
            return value.item()
        return value
        
    def _get_probability(self, mc_results):
        """Safely extract probability value, converting numpy types if needed"""
        prob = mc_results.get('prob_positive', 0)
        if hasattr(prob, "item"):
            return prob.item()
        return prob
        
    def _get_value(self, results, key, default=0):
        """Safely extract a value, converting numpy types if needed"""
        value = results.get(key, default)
        if hasattr(value, "item"):
            return value.item()
        return value