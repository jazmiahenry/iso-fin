import pandas as pd
import yfinance as yf
import datetime
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
import re
import os
import json

logger = logging.getLogger(__name__)

class YahooFinanceDataSource:
    def __init__(self):
        """Initialize the Yahoo Finance data source"""
        self.ticker_mapping = self._load_ticker_mapping()
        self.price_cache = {}  # Cache to store retrieved price data
    
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
                "bitcoin": "BTC-USD",
                "ethereum": "ETH-USD",
                "solana": "SOL-USD",
                "cardano": "ADA-USD",
                "binance": "BNB-USD",
                "ripple": "XRP-USD",
                "dogecoin": "DOGE-USD",
                "polkadot": "DOT-USD",
                "avalanche": "AVAX-USD",
                "cosmos": "ATOM-USD"
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
        
        # If it's a common cryptocurrency, try to guess the ticker
        if asset_type == "crypto" and "-USD" not in asset_name:
            potential_ticker = f"{asset_name.upper()}-USD"
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
        
        return matches
    
    def get_historical_data_for_asset(self, 
                          asset_type: str, 
                          asset_name: str, 
                          period: str = "5y",
                          interval: str = "1d") -> Optional[pd.DataFrame]:
        """
        Get historical price data for an asset
        
        Args:
            asset_type: Type of asset (crypto, traditional, alternative)
            asset_name: Name of the asset
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
            
        Returns:
            DataFrame with historical data or None if not available
        """
        # Get ticker
        ticker = self.get_ticker_for_asset(asset_type, asset_name)
        
        if not ticker:
            return None
        
        # Check cache first
        cache_key = f"{ticker}_{period}_{interval}"
        if cache_key in self.price_cache:
            return self.price_cache[cache_key]
        
        try:
            # Get data from Yahoo Finance
            data = yf.download(ticker, period=period, interval=interval, progress=False)
            
            if data.empty:
                logger.warning(f"No data returned for {ticker}")
                return None
            
            # Add ticker name as column
            data['Ticker'] = ticker
            
            # Store in cache
            self.price_cache[cache_key] = data
            
            return data
        
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {str(e)}")
            return None
    
    def get_returns_and_volatility(self, 
                                 asset_type: str, 
                                 asset_name: str,
                                 period: str = "5y") -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate annualized returns and volatility for an asset
        
        Args:
            asset_type: Type of asset
            asset_name: Name of asset
            period: Time period for calculation
            
        Returns:
            Tuple of (annualized_return, annualized_volatility) or (None, None) if data unavailable
        """
        data = self.get_historical_data_for_asset(asset_type, asset_name, period=period)
        
        if data is None or len(data) < 2:
            return None, None
        
        try:
            # Calculate daily returns
            data['Returns'] = data['Adj Close'].pct_change()
            
            # Remove NaN values
            data = data.dropna()
            
            if len(data) < 30:
                logger.warning(f"Insufficient data points for {asset_name} to calculate reliable metrics")
                return None, None
            
            # Calculate annualized return
            total_return = (data['Adj Close'].iloc[-1] / data['Adj Close'].iloc[0]) - 1
            years = len(data) / 252  # Approximate trading days in a year
            annualized_return = (1 + total_return) ** (1 / years) - 1
            
            # Calculate annualized volatility
            daily_volatility = data['Returns'].std()
            annualized_volatility = daily_volatility * (252 ** 0.5)  # Multiply by sqrt of trading days
            
            return annualized_return, annualized_volatility
        
        except Exception as e:
            logger.error(f"Error calculating metrics for {asset_name}: {str(e)}")
            return None, None
    
    def get_correlation_matrix(self, assets: List[Dict[str, str]], period: str = "1y") -> Optional[pd.DataFrame]:
        """
        Calculate correlation matrix for a list of assets
        
        Args:
            assets: List of dictionaries with asset_type and asset_name
            period: Time period for calculation
            
        Returns:
            DataFrame with correlation matrix or None if data unavailable
        """
        # Get tickers for all assets
        tickers = []
        ticker_names = []
        
        for asset in assets:
            asset_type = asset.get('type', '')
            asset_name = asset.get('name', '')
            
            ticker = self.get_ticker_for_asset(asset_type, asset_name)
            if ticker:
                tickers.append(ticker)
                ticker_names.append(f"{asset_type}_{asset_name}")
        
        if not tickers:
            logger.warning("No valid tickers found for correlation matrix")
            return None
        
        try:
            # Download data for all tickers
            data = yf.download(tickers, period=period, interval="1d", progress=False)
            
            if data.empty:
                logger.warning("No data returned for correlation calculation")
                return None
            
            # Extract adjusted close prices
            prices = data['Adj Close']
            
            # Rename columns to asset names
            prices.columns = ticker_names
            
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
            # Calculate start date
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=days)
            
            # Convert tickers list to a space-separated string if it's a list
            # This is needed because yfinance expects a string for multiple tickers
            if isinstance(tickers, list):
                ticker_str = " ".join(tickers)
            else:
                ticker_str = tickers
                
            logger.info(f"Downloading data for tickers: {ticker_str}")
            
            # Download data
            data = yf.download(
                ticker_str, 
                start=start_date, 
                end=end_date, 
                progress=False
            )
            
            # Check if we're dealing with a single ticker
            if isinstance(tickers, list) and len(tickers) == 1 or isinstance(tickers, str):
                # For a single ticker, yfinance returns a DataFrame without a MultiIndex
                # Simply return the Adjusted Close 
                if 'Adj Close' in data.columns:
                    # Add ticker name as column name for consistency
                    ticker_name = tickers[0] if isinstance(tickers, list) else tickers
                    result = pd.DataFrame(data['Adj Close'])
                    result.columns = [ticker_name]
                    return result
                else:
                    logger.warning(f"No 'Adj Close' column found in data for {tickers}")
                    return pd.DataFrame()
            else:
                # For multiple tickers, we already have a MultiIndex with ticker names
                # Just extract the Adjusted Close prices
                if 'Adj Close' in data.columns:
                    return data['Adj Close']
                else:
                    logger.warning("No 'Adj Close' column found in data")
                    return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching historical data: {str(e)}")
            # Return empty DataFrame with correct structure
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
            ticker_obj = yf.Ticker(ticker)
            ticker_info = ticker_obj.info
            
            if "regularMarketPrice" in ticker_info:
                return ticker_info["regularMarketPrice"]
            elif "lastPrice" in ticker_info:
                return ticker_info["lastPrice"]
            elif "previousClose" in ticker_info:
                return ticker_info["previousClose"]
            else:
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
            ticker_obj = yf.Ticker(ticker)
            return ticker_obj.info
        except Exception as e:
            logger.error(f"Error fetching ticker info for {ticker}: {str(e)}")
            return None