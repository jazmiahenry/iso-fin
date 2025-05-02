"""
Financial agent module that uses a finetuned GPT model to provide financial advice
and calculations using the fin_tools toolkit.
"""

import os
import json
import inspect
import logging
import openai
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

from iso_financial_mvp.toolkit import fin_tools
from iso_financial_mvp.toolkit import fin_data_tools

load_dotenv()

class FinancialGPTAgent:
    """
    Financial agent using a finetuned GPT model to provide financial analysis
    and calculations using the fin_tools toolkit.
    """
    
    def __init__(
        self, 
        model_name: str = "ft:gpt-4.1-2025-04-14:isoai:iso-financial-advisor-v1:BRuCm1OJ"
    ):
        """
        Initialize the financial GPT agent.
        
        Args:
            model_name: Finetuned model name to use
        """
        # Set API key
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable must be set")
            
        # Set OpenAI API key
        openai.api_key = self.api_key
        
        # Set organization ID if available
        org_id = os.environ.get("OPENAI_ORG_ID")
        if org_id:
            openai.organization = org_id
            print(f"Using OpenAI organization: {org_id}")
        
        # Set model name
        self.model = model_name
        
        # Register essential financial tools (limited to 20 most important ones)
        self.tools = self._register_tools()
    
    def _register_tools(self) -> List[Dict[str, Any]]:
        """Register essential financial tools and data tools for function calling."""
        # Core financial calculation tools
        essential_tools = [
            "portfolio_return", "sharpe_ratio", "beta", "alpha",
            "volatility", "value_at_risk", "maximum_drawdown", 
            "present_value", "future_value", "npv", "irr",
            "compound_annual_growth_rate", "annualized_return", 
            "correlation", "treynor_ratio", "sortino_ratio",
            "minimum_variance_portfolio", "expected_return", "mean_return"
        ]
        
        # Data and simulation tools
        data_tools = [
            "get_historical_data", "get_asset_info",
            "simulate_portfolio", "analyze_portfolio_risk",
            "optimize_portfolio", "analyze_scenario",
            "rank_scenarios_for_portfolio", "get_available_scenarios"
        ]
        
        # Get functions from fin_tools and fin_data_tools
        fin_tool_functions = dict(inspect.getmembers(fin_tools, inspect.isfunction))
        fin_data_functions = dict(inspect.getmembers(fin_data_tools, inspect.isfunction))
        
        # Create tools list
        tools = []
        
        # Register core financial tools
        for name in essential_tools:
            if name in fin_tool_functions:
                func = fin_tool_functions[name]
                
                # Get function signature
                sig = inspect.signature(func)
                docstring = inspect.getdoc(func) or ""
                
                # Create parameter schema
                parameters = {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
                
                for param_name, param in sig.parameters.items():
                    # Determine parameter type
                    param_type = "string"
                    if param.annotation == float:
                        param_type = "number"
                    elif param.annotation == int:
                        param_type = "integer"
                    elif param.annotation == bool:
                        param_type = "boolean"
                    elif param.annotation == List or getattr(param.annotation, "__origin__", None) == list:
                        param_type = "array"
                        # For arrays, we need to specify the items type (required by OpenAI API)
                        # Try to get the list item type
                        args = getattr(param.annotation, "__args__", None)
                        item_type = "string"  # Default
                        if args and args[0] == str:
                            item_type = "string"
                        elif args and args[0] == int:
                            item_type = "integer"
                        elif args and args[0] == float:
                            item_type = "number"
                        
                        parameters["properties"][param_name] = {
                            "type": param_type,
                            "description": param_name.replace("_", " "),
                            "items": {"type": item_type}
                        }
                        continue  # Skip the standard parameter addition below
                    elif param.annotation == Dict or getattr(param.annotation, "__origin__", None) == dict:
                        param_type = "object"
                        # For dictionaries, we need to add properties (required by OpenAI API)
                        parameters["properties"][param_name] = {
                            "type": param_type,
                            "description": param_name.replace("_", " "),
                            "additionalProperties": True
                        }
                        continue  # Skip the standard parameter addition below
                    
                    # Add parameter to schema
                    parameters["properties"][param_name] = {
                        "type": param_type,
                        "description": param_name.replace("_", " ")
                    }
                    
                    # Mark required parameters
                    if param.default == inspect.Parameter.empty:
                        parameters["required"].append(param_name)
                
                # Add tool to list
                tools.append({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": docstring.split("\n")[0] if docstring else f"Calculate {name.replace('_', ' ')}",
                        "parameters": parameters
                    }
                })
        
        # Register data and simulation tools
        for name in data_tools:
            if name in fin_data_functions:
                func = fin_data_functions[name]
                
                # Get function signature
                sig = inspect.signature(func)
                docstring = inspect.getdoc(func) or ""
                
                # Create parameter schema
                parameters = {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
                
                for param_name, param in sig.parameters.items():
                    # Determine parameter type
                    param_type = "string"
                    if param.annotation == float:
                        param_type = "number"
                    elif param.annotation == int:
                        param_type = "integer"
                    elif param.annotation == bool:
                        param_type = "boolean"
                    elif param.annotation == List or getattr(param.annotation, "__origin__", None) == list:
                        param_type = "array"
                        # For arrays, we need to specify the items type (required by OpenAI API)
                        # Try to get the list item type
                        args = getattr(param.annotation, "__args__", None)
                        item_type = "string"  # Default
                        if args and args[0] == str:
                            item_type = "string"
                        elif args and args[0] == int:
                            item_type = "integer"
                        elif args and args[0] == float:
                            item_type = "number"
                        
                        parameters["properties"][param_name] = {
                            "type": param_type,
                            "description": param_name.replace("_", " "),
                            "items": {"type": item_type}
                        }
                        continue  # Skip the standard parameter addition below
                    elif param.annotation == Dict or getattr(param.annotation, "__origin__", None) == dict:
                        param_type = "object"
                        # For dictionaries, we need to add properties (required by OpenAI API)
                        parameters["properties"][param_name] = {
                            "type": param_type,
                            "description": param_name.replace("_", " "),
                            "additionalProperties": True
                        }
                        continue  # Skip the standard parameter addition below
                    
                    # Add parameter to schema
                    parameters["properties"][param_name] = {
                        "type": param_type,
                        "description": param_name.replace("_", " ")
                    }
                    
                    # Mark required parameters
                    if param.default == inspect.Parameter.empty:
                        parameters["required"].append(param_name)
                
                # Add tool to list
                tools.append({
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": docstring.split("\n")[0] if docstring else f"Use {name.replace('_', ' ')}",
                        "parameters": parameters
                    }
                })
        
        return tools
    
    def _execute_tool(self, name: str, arguments: Dict[str, Any]) -> str:
        """Execute a financial tool or data tool and return the result as a string."""
        try:
            # First, try to find the function in fin_tools
            if hasattr(fin_tools, name):
                func = getattr(fin_tools, name)
                result = func(**arguments)
            # If not found, try fin_data_tools
            elif hasattr(fin_data_tools, name):
                func = getattr(fin_data_tools, name)
                result = func(**arguments)
            else:
                return f"Error: Function {name} not found in available tools"
            
            # Format the result
            if isinstance(result, (dict, list)):
                return json.dumps(result, indent=2)
            else:
                return str(result)
                
        except Exception as e:
            return f"Error executing {name}: {str(e)}"
    
    def answer_question(
        self, 
        question: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a financial query using the model with tool support.
        
        Args:
            question: User's financial question
            context: Optional context information
            
        Returns:
            Dictionary with response and calculations
        """
        # Create system prompt
        system_prompt = """
        You are a financial advisor with expertise in investment analysis, 
        portfolio management, risk assessment, and financial planning. Use the
        financial calculation tools and data analysis tools to provide precise answers.
        
        You have access to two categories of tools:
        
        1. Financial Calculation Tools:
        - Tools like portfolio_return, sharpe_ratio, beta, alpha, volatility, etc.
        - Use these for calculations and financial metrics
        
        2. Data and Simulation Tools:
        - get_historical_data: Get historical price data for tickers
        - get_asset_info: Get detailed information about an asset
        - simulate_portfolio: Run Monte Carlo simulations on a portfolio
        - analyze_portfolio_risk: Get detailed risk metrics (VaR, CVaR, etc.)
        - optimize_portfolio: Get risk-parity optimized weights for a portfolio
        - analyze_scenario: Test how a portfolio would perform in specific scenarios
        - rank_scenarios_for_portfolio: Get most likely scenarios affecting a portfolio
        - get_available_scenarios: List available economic scenarios to analyze
        
        When answering questions about portfolios, market data, or performance forecasts,
        utilize the data and simulation tools to provide evidence-based answers.
        """
        
        # Add context to prompt if provided
        if context:
            context_str = "\nAdditional context for this query:\n"
            for key, value in context.items():
                context_str += f"- {key}: {value}\n"
            system_prompt += context_str
        
        try:
            # First API call with tools
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ]
            
            response = openai.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
                max_tokens=1000,
                temperature=0.2
            )
            
            # Get response and handle tool calls
            response_message = response.choices[0].message
            calculations = []
            
            # Check if model used any tools
            tool_calls = response_message.tool_calls
            if tool_calls:
                # Process each tool call
                tool_messages = []
                for tool_call in tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    # Execute the function
                    function_response = self._execute_tool(function_name, function_args)
                    
                    # Record calculation for return value
                    calculations.append({
                        "tool": function_name,
                        "inputs": function_args,
                        "result": function_response
                    })
                    
                    # Add to messages for second API call
                    tool_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": function_response
                    })
                
                # Second API call with tool results
                messages.append(response_message)
                messages.extend(tool_messages)
                
                final_response = openai.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=1000,
                    temperature=0.2
                )
                
                answer = final_response.choices[0].message.content
            else:
                # No tools used, return original response
                answer = response_message.content
            
            return {
                "response": answer,
                "calculations": calculations
            }
        
        except Exception as e:
            return {
                "response": f"I'm sorry, I encountered an error: {str(e)}",
                "calculations": []
            }