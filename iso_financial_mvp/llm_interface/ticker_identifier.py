import os
import re
import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import yfinance as yf

# Local imports
from iso_financial_mvp.data_sources.polygon_client import PolygonDataSource

# Setup logger
logger = logging.getLogger(__name__)

class TickerIdentifier:
    def __init__(self, data_source: Optional[PolygonDataSource] = None):
        """
        Initialize ticker identifier to map asset descriptions to ticker symbols
        
        Args:
            data_source: Optional PolygonDataSource instance
        """
        self.data_source = data_source or PolygonDataSource()
        self.ticker_patterns = self._load_ticker_patterns()
    
    def _load_ticker_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load ticker identification patterns
        
        Returns:
            Dictionary of patterns for asset types and names
        """
        # Common identification patterns for different asset types
        return {
            "crypto": [
                {"pattern": r"(?i)bitcoin|btc", "name": "bitcoin", "type": "crypto"},
                {"pattern": r"(?i)ethereum|eth", "name": "ethereum", "type": "crypto"},
                {"pattern": r"(?i)solana|sol", "name": "solana", "type": "crypto"},
                {"pattern": r"(?i)cardano|ada", "name": "cardano", "type": "crypto"},
                {"pattern": r"(?i)binance|bnb", "name": "binance", "type": "crypto"},
                {"pattern": r"(?i)ripple|xrp", "name": "ripple", "type": "crypto"},
                {"pattern": r"(?i)dogecoin|doge", "name": "dogecoin", "type": "crypto"},
                {"pattern": r"(?i)polkadot|dot", "name": "polkadot", "type": "crypto"},
                {"pattern": r"(?i)polygon|matic", "name": "polygon", "type": "crypto"},
                {"pattern": r"(?i)litecoin|ltc", "name": "litecoin", "type": "crypto"},
                {"pattern": r"(?i)chainlink|link", "name": "chainlink", "type": "crypto"},
                {"pattern": r"(?i)avalanche|avax", "name": "avalanche", "type": "crypto"},
                {"pattern": r"(?i)uniswap|uni", "name": "uniswap", "type": "crypto"},
                {"pattern": r"(?i)cosmos|atom", "name": "cosmos", "type": "crypto"},
                {"pattern": r"(?i)usdc|usd coin", "name": "stablecoin", "type": "crypto"}
            ],
            "traditional": [
                {"pattern": r"(?i)s&p\s*500|sp\s*500|spy", "name": "us_stocks", "type": "traditional"},
                {"pattern": r"(?i)nasdaq|qqq", "name": "tech_stocks", "type": "traditional"},
                {"pattern": r"(?i)dow\s*jones|dia", "name": "us_stocks", "type": "traditional"},
                {"pattern": r"(?i)u\.?s\.?\s*stocks?|american\s*stocks?", "name": "us_stocks", "type": "traditional"},
                {"pattern": r"(?i)global\s*stocks?|international\s*stocks?|world\s*stocks?", "name": "global_stocks", "type": "traditional"},
                {"pattern": r"(?i)tech\s*stocks?|technology\s*stocks?", "name": "tech_stocks", "type": "traditional"},
                {"pattern": r"(?i)bond|fixed\s*income", "name": "bonds", "type": "traditional"},
                {"pattern": r"(?i)treasury\s*bonds?|government\s*bonds?", "name": "treasury_bonds", "type": "traditional"},
                {"pattern": r"(?i)gold|gld", "name": "gold", "type": "traditional"},
                {"pattern": r"(?i)commodit(y|ies)", "name": "commodities", "type": "traditional"},
                {"pattern": r"(?i)real\s*estate\s*investment\s*trust|reit", "name": "real_estate", "type": "traditional"},
                {"pattern": r"(?i)emerging\s*markets?", "name": "emerging_markets", "type": "traditional"},
                {"pattern": r"(?i)small\s*cap", "name": "small_cap", "type": "traditional"},
                {"pattern": r"(?i)large\s*cap", "name": "large_cap", "type": "traditional"},
                {"pattern": r"(?i)energy\s*sector", "name": "energy", "type": "traditional"},
                {"pattern": r"(?i)financials?|banks?", "name": "financials", "type": "traditional"},
                {"pattern": r"(?i)healthcare", "name": "healthcare", "type": "traditional"}
            ],
            "nft": [
                {"pattern": r"(?i)bored\s*ape|bayc", "name": "bored_ape", "type": "nft"},
                {"pattern": r"(?i)cryptopunks?", "name": "cryptopunks", "type": "nft"},
                {"pattern": r"(?i)art\s*blocks", "name": "art_blocks", "type": "nft"},
                {"pattern": r"(?i)nft\s*index|nft\s*market", "name": "nft_index", "type": "nft"}
            ],
            "real_estate": [
                {"pattern": r"(?i)residential\s*real\s*estate", "name": "residential", "type": "real_estate"},
                {"pattern": r"(?i)commercial\s*real\s*estate", "name": "commercial", "type": "real_estate"},
                {"pattern": r"(?i)reit|real\s*estate\s*investment\s*trust", "name": "reit", "type": "real_estate"},
                {"pattern": r"(?i)international\s*real\s*estate", "name": "international", "type": "real_estate"},
                {"pattern": r"(?i)mortgage\s*reit", "name": "mortgage", "type": "real_estate"}
            ],
            "collectible": [
                {"pattern": r"(?i)art\s*collection|art\s*market", "name": "art", "type": "collectible"},
                {"pattern": r"(?i)watches?|luxury\s*watches?", "name": "watches", "type": "collectible"},
                {"pattern": r"(?i)wine\s*collection|fine\s*wine", "name": "wine", "type": "collectible"},
                {"pattern": r"(?i)coins?|rare\s*coins?", "name": "coins", "type": "collectible"},
                {"pattern": r"(?i)trading\s*cards?|sports\s*cards?", "name": "cards", "type": "collectible"},
                {"pattern": r"(?i)memorabilia", "name": "memorabilia", "type": "collectible"}
            ],
            "alternative": [
                {"pattern": r"(?i)crypto\s*index|crypto\s*fund", "name": "crypto_index", "type": "alternative"},
                {"pattern": r"(?i)private\s*equity", "name": "private_equity", "type": "alternative"},
                {"pattern": r"(?i)hedge\s*funds?", "name": "hedge_funds", "type": "alternative"},
                {"pattern": r"(?i)venture\s*capital|vc", "name": "venture_capital", "type": "alternative"},
                {"pattern": r"(?i)luxury\s*goods", "name": "collectibles", "type": "alternative"},
                {"pattern": r"(?i)rare\s*coins", "name": "rare_coins", "type": "alternative"},
                {"pattern": r"(?i)wine\s*investment", "name": "wine", "type": "alternative"},
                {"pattern": r"(?i)luxury\s*watches", "name": "luxury_watches", "type": "alternative"},
                {"pattern": r"(?i)rare\s*books", "name": "rare_books", "type": "alternative"},
                {"pattern": r"(?i)classic\s*cars", "name": "classic_cars", "type": "alternative"}
            ]
        }
    
    def identify_assets_in_query(self, query: str) -> List[Dict[str, Any]]:
        """
        Identify asset types and names mentioned in a query
        
        Args:
            query: Natural language query string
            
        Returns:
            List of identified assets with type and name
        """
        identified_assets = []
        
        # Check for matches in each asset category
        for asset_category, patterns in self.ticker_patterns.items():
            for pattern_info in patterns:
                pattern = pattern_info["pattern"]
                
                if re.search(pattern, query):
                    asset_info = {
                        "type": pattern_info["type"],
                        "name": pattern_info["name"]
                    }
                    
                    # Check if this asset is already identified
                    if not any(
                        asset["type"] == asset_info["type"] and asset["name"] == asset_info["name"] 
                        for asset in identified_assets
                    ):
                        identified_assets.append(asset_info)
        
        # If no assets identified, look for general categories
        if not identified_assets:
            # Check for general asset categories
            if re.search(r"(?i)crypto|cryptocurrency", query):
                identified_assets.append({"type": "crypto", "name": "general"})
            
            if re.search(r"(?i)stocks?|equities", query):
                identified_assets.append({"type": "traditional", "name": "us_stocks"})
            
            if re.search(r"(?i)bonds?|fixed\s*income", query):
                identified_assets.append({"type": "traditional", "name": "bonds"})
            
            if re.search(r"(?i)nfts?|non-fungible", query):
                identified_assets.append({"type": "nft", "name": "general"})
            
            if re.search(r"(?i)real\s*estate", query):
                identified_assets.append({"type": "real_estate", "name": "general"})
            
            if re.search(r"(?i)collectibles?", query):
                identified_assets.append({"type": "collectible", "name": "general"})
        
        # Get tickers for identified assets
        for asset in identified_assets:
            ticker = self.data_source.get_ticker_for_asset(asset["type"], asset["name"])
            if ticker:
                asset["ticker"] = ticker
        
        return identified_assets
    
    def lookup_ticker(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Look up a ticker symbol from text description
        
        Args:
            text: Text description of an asset
            
        Returns:
            Dictionary with asset details including ticker, or None if not found
        """
        # First, check if the text is already a valid ticker
        try:
            ticker_info = yf.Ticker(text.upper()).info
            if "regularMarketPrice" in ticker_info or "previousClose" in ticker_info:
                return {
                    "ticker": text.upper(),
                    "type": "unknown",
                    "name": ticker_info.get("shortName", text.upper())
                }
        except:
            pass
        
        # Check if any defined patterns match
        for asset_category, patterns in self.ticker_patterns.items():
            for pattern_info in patterns:
                pattern = pattern_info["pattern"]
                
                if re.search(pattern, text):
                    ticker = self.data_source.get_ticker_for_asset(
                        pattern_info["type"], pattern_info["name"]
                    )
                    
                    if ticker:
                        return {
                            "ticker": ticker,
                            "type": pattern_info["type"],
                            "name": pattern_info["name"]
                        }
        
        # If no match found, try to search ticker mapping
        search_results = self.data_source.search_ticker(text)
        
        if search_results:
            return search_results[0]  # Return first match
        
        return None
    
    def identify_tickers_for_portfolio(self, portfolio: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Identify ticker symbols for assets in a portfolio
        
        Args:
            portfolio: List of portfolio assets with type and name
            
        Returns:
            Portfolio with ticker symbols added where available
        """
        portfolio_with_tickers = []
        
        for asset in portfolio:
            asset_type = asset.get("type", "")
            asset_name = asset.get("name", "")
            
            # Check if ticker is already included
            if "ticker" not in asset:
                ticker = self.data_source.get_ticker_for_asset(asset_type, asset_name)
                
                if ticker:
                    # Create a copy to avoid modifying the original
                    asset_copy = asset.copy()
                    asset_copy["ticker"] = ticker
                    portfolio_with_tickers.append(asset_copy)
                else:
                    portfolio_with_tickers.append(asset)
            else:
                portfolio_with_tickers.append(asset)
        
        return portfolio_with_tickers
    
    def extract_ticker_from_text(self, text: str) -> List[str]:
        """
        Extract potential ticker symbols from text
        
        Args:
            text: Input text
            
        Returns:
            List of potential ticker symbols
        """
        # Find potential ticker patterns
        # Typical ticker format is 1-5 uppercase letters, or crypto tickers with -USD suffix
        ticker_pattern = r'\b[A-Z]{1,5}\b|\b[A-Z]{1,5}-USD\b'
        potential_tickers = re.findall(ticker_pattern, text.upper())
        
        # Filter out common words that might be mistaken for tickers
        common_words = ["A", "I", "AM", "AN", "AS", "AT", "BE", "BY", "GO", "IF", "IN", "IS", 
                        "IT", "ME", "MY", "NO", "OF", "ON", "OR", "SO", "TO", "UP", "US", "WE"]
        
        filtered_tickers = [t for t in potential_tickers if t not in common_words]
        
        return filtered_tickers
    
    def enhance_portfolio_data(self, portfolio: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Enhance portfolio data with tickers and current prices
        
        Args:
            portfolio: Portfolio asset list
            
        Returns:
            Enhanced portfolio with additional data
        """
        enhanced_portfolio = []
        
        for asset in portfolio:
            asset_type = asset.get("type", "")
            asset_name = asset.get("name", "")
            
            # Create a copy to avoid modifying the original
            enhanced_asset = asset.copy()
            
            # Add ticker if not present
            if "ticker" not in enhanced_asset:
                ticker = self.data_source.get_ticker_for_asset(asset_type, asset_name)
                if ticker:
                    enhanced_asset["ticker"] = ticker
            
            # Get current price
            if "ticker" in enhanced_asset:
                price = self.data_source.get_current_price(asset_type, asset_name)
                if price:
                    enhanced_asset["current_price"] = price
            
            # Get historical performance data
            if "ticker" in enhanced_asset:
                annual_return, annual_volatility = self.data_source.get_returns_and_volatility(
                    asset_type, asset_name
                )
                
                if annual_return is not None:
                    enhanced_asset["annual_return"] = annual_return
                
                if annual_volatility is not None:
                    enhanced_asset["annual_volatility"] = annual_volatility
            
            enhanced_portfolio.append(enhanced_asset)
        
        return enhanced_portfolio