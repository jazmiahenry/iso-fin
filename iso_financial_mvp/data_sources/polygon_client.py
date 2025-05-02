"""
Polygon.io data source for financial data.
"""

import os
import logging
import json
import pandas as pd
import numpy as np
import requests
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)

class PolygonDataSource:
    def __init__(self):
        """Initialize the Polygon.io data source"""
        self.api_key = os.environ.get("POLYGON")
        if not self.api_key:
            raise ValueError("POLYGON API key not found in environment variables")
        
        self.base_url = "https://api.polygon.io"
        self.price_cache = {}  # Cache to store retrieved price data
        self.ticker_mapping = self._load_ticker_mapping()
    
    def _load_ticker_mapping(self) -> Dict[str, Dict[str, str]]:
        """
        Load ticker mapping from file or initialize default mapping
        
        Returns:
            Dictionary mapping asset types and names to tickers
        """
        # Get data directory path
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        ticker_map_path = os.path.join(data_dir, "ticker_mapping.json")
        
        # Try to load ticker mapping
        try:
            if os.path.exists(ticker_map_path):
                with open(ticker_map_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load ticker mapping: {str(e)}")
        
        # Default mapping if file doesn't exist or loading fails
        return {
            "crypto": {
                "bitcoin": "X:BTCUSD",
                "ethereum": "X:ETHUSD",
                "solana": "X:SOLUSD",
                "cardano": "X:ADAUSD",
                "binance": "X:BNBUSD",
                "ripple": "X:XRPUSD",
                "dogecoin": "X:DOGEUSD",
                "polkadot": "X:DOTUSD",
                "avalanche": "X:AVAXUSD",
                "cosmos": "X:ATOMUSD"
            },
            "traditional": {
                "us_stocks": "SPY",  # S&P 500 ETF
                "global_stocks": "ACWI",  # MSCI All Country World Index ETF
                "tech_stocks": "QQQ",  # Nasdaq 100 ETF
                "bonds": "AGG",  # US Aggregate Bond ETF
                "treasury_bonds": "IEF",  # 7-10 Year Treasury Bond ETF
                "gold": "GLD",  # Gold ETF
                "commodities": "DBC",  # Commodity Index ETF
                "real_estate": "VNQ",  # Vanguard Real Estate ETF
                "emerging_markets": "EEM",  # Emerging Markets ETF
                "small_cap": "IWM"  # Russell 2000 ETF
            },
            "alternative": {
                "crypto_index": "BITW",  # Bitwise 10 Crypto Index Fund
                "private_equity": "PSP",  # Global Listed Private Equity ETF
                "hedge_funds": "QAI",  # IQ Hedge Multi-Strategy Tracker ETF
                "venture_capital": "ARKK", # Close approximation through innovation ETF
                "art": "ARTNA", # Art index proxy
                "collectibles": "COIN" # Collectibles marketplace proxy
            }
        }
    
    def save_ticker_mapping(self) -> None:
        """Save ticker mapping to file"""
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        ticker_map_path = os.path.join(data_dir, "ticker_mapping.json")
        
        try:
            with open(ticker_map_path, 'w') as f:
                json.dump(self.ticker_mapping, f, indent=2)
            logger.info("Ticker mapping saved successfully")
        except Exception as e:
            logger.error(f"Failed to save ticker mapping: {str(e)}")
    
    def add_ticker_mapping(self, asset_type: str, asset_name: str, ticker: str) -> None:
        """
        Add a new ticker mapping
        
        Args:
            asset_type: Type of asset (crypto, traditional, alternative)
            asset_name: Name of the asset
            ticker: Ticker symbol for the asset
        """
        if asset_type not in self.ticker_mapping:
            self.ticker_mapping[asset_type] = {}
        
        self.ticker_mapping[asset_type][asset_name] = ticker
        
        # Save updated mapping
        self.save_ticker_mapping()
    
    def get_ticker_for_asset(self, asset_type: str, asset_name: str) -> Optional[str]:
        """
        Get the ticker symbol for a given asset
        
        Args:
            asset_type: Type of asset (crypto, traditional, alternative)
            asset_name: Name of the asset
            
        Returns:
            Ticker symbol or None if not found
        """
        # Check if asset type exists in mapping
        if asset_type in self.ticker_mapping:
            # Check if asset name exists in the asset type mapping
            if asset_name in self.ticker_mapping[asset_type]:
                return self.ticker_mapping[asset_type][asset_name]
        
        # Try to look for partial matches
        if asset_type in self.ticker_mapping:
            for name, ticker in self.ticker_mapping[asset_type].items():
                if asset_name.lower() in name.lower() or name.lower() in asset_name.lower():
                    logger.info(f"Using partial match: {name} -> {ticker} for {asset_name}")
                    return ticker
        
        # If it's a cryptocurrency, try to use the Polygon.io format
        if asset_type == "crypto":
            potential_ticker = f"X:{asset_name.upper()}USD"
            logger.info(f"Using guessed crypto ticker: {potential_ticker}")
            return potential_ticker
        
        logger.warning(f"No ticker found for {asset_type}/{asset_name}")
        return None
    
    def search_ticker(self, query: str) -> List[Dict[str, str]]:
        """
        Search for ticker based on a query string
        
        Args:
            query: Search query
            
        Returns:
            List of potential ticker matches
        """
        matches = []
        query_lower = query.lower()
        
        # Search through all asset types and names
        for asset_type, assets in self.ticker_mapping.items():
            for asset_name, ticker in assets.items():
                if query_lower in asset_name.lower() or query_lower in ticker.lower():
                    matches.append({
                        "asset_type": asset_type,
                        "asset_name": asset_name,
                        "ticker": ticker
                    })
        
        # If no matches found in our mapping, try to search Polygon.io
        if not matches:
            try:
                endpoint = f"/v3/reference/tickers?search={query}&active=true&limit=10"
                response = self._make_api_request(endpoint)
                
                if response and "results" in response:
                    for result in response["results"]:
                        matches.append({
                            "asset_type": "traditional",
                            "asset_name": result.get("name", ""),
                            "ticker": result.get("ticker", "")
                        })
            except Exception as e:
                logger.error(f"Error searching Polygon.io: {str(e)}")
        
        return matches
    
    def _make_api_request(self, endpoint: str) -> Optional[Dict]:
        """
        Make a request to the Polygon.io API
        
        Args:
            endpoint: API endpoint to request
            
        Returns:
            Response data or None if failed
        """
        url = f"{self.base_url}{endpoint}"
        
        # Add API key to URL if needed
        if "?" in url:
            url += f"&apiKey={self.api_key}"
        else:
            url += f"?apiKey={self.api_key}"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"API request failed: {str(e)}")
            return None
    
    def _format_date(self, date: datetime) -> str:
        """Format date for Polygon.io API"""
        return date.strftime("%Y-%m-%d")
    
    def get_historical_data_for_asset(self, 
                          asset_type: str, 
                          asset_name: str, 
                          days: int = 365,
                          multiplier: int = 1,
                          timespan: str = "day") -> Optional[pd.DataFrame]:
        """
        Get historical price data for an asset
        
        Args:
            asset_type: Type of asset (crypto, traditional, alternative)
            asset_name: Name of the asset
            days: Number of days of historical data to retrieve
            multiplier: Time multiplier for the timespan
            timespan: Time unit (minute, hour, day, week, month, quarter, year)
            
        Returns:
            DataFrame with historical data or None if not available
        """
        # Get ticker
        ticker = self.get_ticker_for_asset(asset_type, asset_name)
        
        if not ticker:
            return None
        
        # Check cache first
        cache_key = f"{ticker}_{days}_{multiplier}_{timespan}"
        if cache_key in self.price_cache:
            return self.price_cache[cache_key]
        
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Format dates for API
            from_date = self._format_date(start_date)
            to_date = self._format_date(end_date)
            
            # Construct API endpoint
            if ":" in ticker:  # Handle crypto tickers differently
                # For crypto, we need to use the /v2/aggs endpoint
                endpoint = f"/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
            else:
                # For stocks and ETFs, use the same endpoint
                endpoint = f"/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
            
            # Make API request
            response = self._make_api_request(endpoint)
            
            if not response or "results" not in response:
                logger.warning(f"No data returned for {ticker}")
                return None
            
            # Create DataFrame from results
            results = response["results"]
            if not results:
                logger.warning(f"Empty results for {ticker}")
                return None
            
            # Create DataFrame
            df = pd.DataFrame(results)
            
            # Rename columns to match Yahoo Finance format for compatibility
            column_mapping = {
                "o": "Open",
                "h": "High",
                "l": "Low",
                "c": "Close",
                "v": "Volume",
                "t": "timestamp"
            }
            df = df.rename(columns=column_mapping)
            
            # Convert timestamp to datetime
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df = df.set_index("timestamp")
            
            # Add "Adj Close" column for compatibility with Yahoo Finance
            df["Adj Close"] = df["Close"]
            
            # Add ticker name as column
            df["Ticker"] = ticker
            
            # Store in cache
            self.price_cache[cache_key] = df
            
            return df
        
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {str(e)}")
            return None
    
    def get_returns_and_volatility(self, 
                               asset_type: str, 
                               asset_name: str,
                               days: int = 365) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate annualized returns and volatility for an asset
        
        Args:
            asset_type: Type of asset
            asset_name: Name of asset
            days: Number of days of historical data to use
            
        Returns:
            Tuple of (annualized_return, annualized_volatility) or (None, None) if data unavailable
        """
        data = self.get_historical_data_for_asset(asset_type, asset_name, days=days)
        
        if data is None or len(data) < 2:
            return None, None
        
        try:
            # Calculate daily returns
            data['Returns'] = data['Close'].pct_change()
            
            # Remove NaN values
            data = data.dropna()
            
            if len(data) < 30:
                logger.warning(f"Insufficient data points for {asset_name} to calculate reliable metrics")
                return None, None
            
            # Calculate annualized return
            total_return = (data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1
            years = len(data) / 252  # Approximate trading days in a year
            annualized_return = (1 + total_return) ** (1 / years) - 1
            
            # Calculate annualized volatility
            daily_volatility = data['Returns'].std()
            annualized_volatility = daily_volatility * (252 ** 0.5)  # Multiply by sqrt of trading days
            
            return annualized_return, annualized_volatility
        
        except Exception as e:
            logger.error(f"Error calculating metrics for {asset_name}: {str(e)}")
            return None, None
    
    def get_correlation_matrix(self, assets: List[Dict[str, str]], days: int = 365) -> Optional[pd.DataFrame]:
        """
        Calculate correlation matrix for a list of assets
        
        Args:
            assets: List of dictionaries with asset_type and asset_name
            days: Number of days of historical data to use
            
        Returns:
            DataFrame with correlation matrix or None if data unavailable
        """
        # Get data for all assets
        asset_data = {}
        
        for asset in assets:
            asset_type = asset.get('type', '')
            asset_name = asset.get('name', '')
            
            data = self.get_historical_data_for_asset(asset_type, asset_name, days=days)
            if data is not None and not data.empty:
                asset_key = f"{asset_type}_{asset_name}"
                asset_data[asset_key] = data['Close']
        
        if not asset_data:
            logger.warning("No valid data found for correlation matrix")
            return None
        
        try:
            # Create DataFrame with all asset prices
            prices = pd.DataFrame(asset_data)
            
            # Calculate returns
            returns = prices.pct_change().dropna()
            
            # Calculate correlation matrix
            correlation_matrix = returns.corr()
            
            return correlation_matrix
        
        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {str(e)}")
            return None
    
    def get_asset_data_for_simulation(self, assets: List[Dict[str, str]]) -> Dict[str, Dict[str, float]]:
        """
        Get returns, volatility, and correlation data for a list of assets for simulation
        
        Args:
            assets: List of dictionaries with asset_type and asset_name
            
        Returns:
            Dictionary with asset data for simulation
        """
        asset_data = {}
        
        # Get returns and volatility for each asset
        for asset in assets:
            asset_type = asset.get('type', '')
            asset_name = asset.get('name', '')
            
            ann_return, ann_volatility = self.get_returns_and_volatility(asset_type, asset_name)
            
            if ann_return is not None and ann_volatility is not None:
                asset_data[f"{asset_type}_{asset_name}"] = {
                    "annual_return": ann_return,
                    "annual_volatility": ann_volatility
                }
            else:
                # Use default values if data not available
                logger.warning(f"Using default values for {asset_type}_{asset_name}")
                
                # Default returns and volatility based on asset type
                if asset_type == "crypto":
                    ann_return = 0.25
                    ann_volatility = 0.70
                elif asset_type == "traditional":
                    if "stock" in asset_name:
                        ann_return = 0.10
                        ann_volatility = 0.18
                    elif "bond" in asset_name:
                        ann_return = 0.04
                        ann_volatility = 0.05
                    else:
                        ann_return = 0.08
                        ann_volatility = 0.15
                elif asset_type == "nft" or asset_type == "collectible":
                    ann_return = 0.15
                    ann_volatility = 0.50
                else:
                    ann_return = 0.08
                    ann_volatility = 0.15
                
                asset_data[f"{asset_type}_{asset_name}"] = {
                    "annual_return": ann_return,
                    "annual_volatility": ann_volatility
                }
        
        # Get correlation matrix
        correlation_matrix = self.get_correlation_matrix(assets)
        
        if correlation_matrix is not None:
            asset_data["correlation_matrix"] = correlation_matrix.to_dict()
        
        return asset_data
    
    def get_historical_data(self, tickers: List[str], days: int = 365) -> pd.DataFrame:
        """
        Get historical price data for multiple tickers
        
        Args:
            tickers: List of ticker symbols
            days: Number of days of history to retrieve
            
        Returns:
            DataFrame with historical data
        """
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Get data for each ticker
            result_dfs = {}
            
            for ticker in tickers:
                # Format dates for API
                from_date = self._format_date(start_date)
                to_date = self._format_date(end_date)
                
                # Construct API endpoint
                if ":" in ticker:  # Handle crypto tickers differently
                    endpoint = f"/v2/aggs/ticker/{ticker}/range/1/day/{from_date}/{to_date}"
                else:
                    endpoint = f"/v2/aggs/ticker/{ticker}/range/1/day/{from_date}/{to_date}"
                
                # Make API request
                response = self._make_api_request(endpoint)
                
                if response and "results" in response:
                    # Create DataFrame from results
                    results = response["results"]
                    if results:
                        df = pd.DataFrame(results)
                        
                        # Rename columns to match Yahoo Finance format
                        column_mapping = {
                            "o": "Open",
                            "h": "High",
                            "l": "Low",
                            "c": "Close",
                            "v": "Volume",
                            "t": "timestamp"
                        }
                        df = df.rename(columns=column_mapping)
                        
                        # Convert timestamp to datetime
                        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                        df = df.set_index("timestamp")
                        
                        # Add Adjusted Close column for compatibility
                        df["Adj Close"] = df["Close"]
                        
                        # Store in result dictionary
                        result_dfs[ticker] = df["Adj Close"]
            
            # Combine all DataFrames
            if result_dfs:
                result = pd.DataFrame(result_dfs)
                return result
            else:
                logger.warning("No data found for any tickers")
                return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {str(e)}")
            return pd.DataFrame()
    
    def get_current_price(self, asset_type: str, asset_name: str) -> Optional[float]:
        """
        Get current price for an asset
        
        Args:
            asset_type: Type of asset
            asset_name: Name of asset
            
        Returns:
            Current price or None if unavailable
        """
        ticker = self.get_ticker_for_asset(asset_type, asset_name)
        
        if not ticker:
            return None
        
        try:
            # Use the /v2/last endpoint to get the last trade
            endpoint = f"/v2/last/trade/{ticker}"
            response = self._make_api_request(endpoint)
            
            if response and "results" in response:
                return response["results"].get("p")  # The price is in the "p" field
            else:
                # If that fails, try to get the last closing price
                data = self.get_historical_data_for_asset(asset_type, asset_name, days=5)
                if data is not None and not data.empty:
                    return data["Close"].iloc[-1]
            
            return None
        
        except Exception as e:
            logger.error(f"Error fetching current price for {ticker}: {str(e)}")
            return None
    
    def get_ticker_info(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information for a ticker
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            Dictionary with ticker information or None if unavailable
        """
        try:
            # Use the /v3/reference/tickers endpoint
            endpoint = f"/v3/reference/tickers/{ticker}"
            response = self._make_api_request(endpoint)
            
            if response and "results" in response:
                return response["results"]
            
            return None
        except Exception as e:
            logger.error(f"Error fetching ticker info for {ticker}: {str(e)}")
            return None