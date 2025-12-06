"""
FinnHub Data Client via n8n Webhook
Fetches real-time quotes, fundamentals, and analyst ratings from n8n webhook.
"""

import os
import requests
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging

# Load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

# n8n Webhook configuration (from .env or defaults)
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL")
N8N_AUTH_HEADER = os.getenv("N8N_AUTH_HEADER")
N8N_AUTH_VALUE = os.getenv("N8N_AUTH_VALUE")


@dataclass
class TickerData:
    """Container for ticker data from n8n webhook."""
    ticker: str
    current_price: Optional[float] = None
    change: Optional[float] = None
    change_percent: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    open: Optional[float] = None
    previous_close: Optional[float] = None
    
    # Fundamental metrics
    pe_ratio: Optional[float] = None
    pe_forward: Optional[float] = None
    pb_ratio: Optional[float] = None
    ps_ratio: Optional[float] = None
    eps: Optional[float] = None
    eps_growth: Optional[float] = None
    revenue_growth: Optional[float] = None
    profit_margin: Optional[float] = None
    gross_margin: Optional[float] = None
    roe: Optional[float] = None
    roa: Optional[float] = None
    debt_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    beta: Optional[float] = None
    market_cap: Optional[float] = None
    week52_high: Optional[float] = None
    week52_low: Optional[float] = None
    
    # Analyst ratings
    analyst_buy: int = 0
    analyst_hold: int = 0
    analyst_sell: int = 0
    analyst_strong_buy: int = 0
    analyst_strong_sell: int = 0
    
    @property
    def analyst_score(self) -> float:
        """Calculate analyst score (-1 to +1)."""
        total = self.analyst_buy + self.analyst_hold + self.analyst_sell + \
                self.analyst_strong_buy + self.analyst_strong_sell
        if total == 0:
            return 0
        
        score = (self.analyst_strong_buy * 2 + self.analyst_buy * 1 + 
                 self.analyst_hold * 0 + 
                 self.analyst_sell * -1 + self.analyst_strong_sell * -2)
        return score / (total * 2)  # Normalize to -1 to +1
    
    @property
    def value_score(self) -> float:
        """Calculate value score based on fundamentals."""
        score = 0
        count = 0
        
        # P/E ratio (lower is better, but not negative)
        if self.pe_ratio and self.pe_ratio > 0:
            if self.pe_ratio < 15:
                score += 1.0
            elif self.pe_ratio < 25:
                score += 0.5
            elif self.pe_ratio > 40:
                score -= 0.5
            count += 1
        
        # Profit margin (higher is better)
        if self.profit_margin:
            if self.profit_margin > 0.20:
                score += 1.0
            elif self.profit_margin > 0.10:
                score += 0.5
            elif self.profit_margin < 0:
                score -= 1.0
            count += 1
        
        # ROE (higher is better)
        if self.roe:
            if self.roe > 0.20:
                score += 1.0
            elif self.roe > 0.10:
                score += 0.5
            elif self.roe < 0:
                score -= 1.0
            count += 1
        
        return score / count if count > 0 else 0


class FinnHubClient:
    """Client to fetch data from n8n webhook."""
    
    def __init__(self, webhook_url: str = N8N_WEBHOOK_URL):
        self.webhook_url = webhook_url
        self.headers = {
            N8N_AUTH_HEADER: N8N_AUTH_VALUE,
            "Content-Type": "application/json"
        }
        self._cache: Dict[str, TickerData] = {}
        self._last_fetch = None
    
    def fetch_all(self, timeout: int = 60) -> Dict[str, TickerData]:
        """
        Fetch all ticker data from n8n webhook.
        
        Returns:
            Dictionary mapping ticker symbols to TickerData objects
        """
        try:
            response = requests.get(
                self.webhook_url,
                headers=self.headers,
                timeout=timeout
            )
            response.raise_for_status()
            
            data = response.json()
            
            # Handle format: [{success, data: [...]}] - array containing wrapper object
            if isinstance(data, list) and len(data) > 0:
                wrapper = data[0]
                if isinstance(wrapper, dict) and "data" in wrapper:
                    items = wrapper.get("data", [])
                else:
                    items = data  # Direct array of items
            elif isinstance(data, dict):
                items = data.get("data", [])
            else:
                logger.warning(f"Unexpected response type: {type(data)}")
                return {}
            
            result = {}
            for item in items:
                ticker = item.get("ticker")
                if not ticker:
                    continue
                
                quote = item.get("quote", {})
                metrics = item.get("metrics", {})
                analyst = item.get("analyst", {})
                
                result[ticker] = TickerData(
                    ticker=ticker,
                    current_price=quote.get("current_price"),
                    change=quote.get("change"),
                    change_percent=quote.get("change_percent"),
                    high=quote.get("high"),
                    low=quote.get("low"),
                    open=quote.get("open"),
                    previous_close=quote.get("previous_close"),
                    pe_ratio=metrics.get("pe_ratio"),
                    pe_forward=metrics.get("pe_forward"),
                    pb_ratio=metrics.get("pb_ratio"),
                    ps_ratio=metrics.get("ps_ratio"),
                    eps=metrics.get("eps"),
                    eps_growth=metrics.get("eps_growth"),
                    revenue_growth=metrics.get("revenue_growth"),
                    profit_margin=metrics.get("profit_margin"),
                    gross_margin=metrics.get("gross_margin"),
                    roe=metrics.get("roe"),
                    roa=metrics.get("roa"),
                    debt_equity=metrics.get("debt_equity"),
                    current_ratio=metrics.get("current_ratio"),
                    dividend_yield=metrics.get("dividend_yield"),
                    beta=metrics.get("beta"),
                    market_cap=metrics.get("market_cap"),
                    week52_high=metrics.get("week52_high"),
                    week52_low=metrics.get("week52_low"),
                    analyst_buy=analyst.get("buy", 0),
                    analyst_hold=analyst.get("hold", 0),
                    analyst_sell=analyst.get("sell", 0),
                    analyst_strong_buy=analyst.get("strong_buy", 0),
                    analyst_strong_sell=analyst.get("strong_sell", 0),
                )
            
            self._cache = result
            logger.info(f"Fetched {len(result)} tickers from n8n webhook")
            return result
            
        except requests.RequestException as e:
            logger.error(f"Failed to fetch from n8n webhook: {e}")
            return self._cache  # Return cached data on error
    
    def get_ticker(self, ticker: str) -> Optional[TickerData]:
        """Get data for a specific ticker."""
        if ticker in self._cache:
            return self._cache[ticker]
        
        # Fetch fresh data if not cached
        self.fetch_all()
        return self._cache.get(ticker)
    
    def get_fundamental_dict(self, ticker: str) -> Dict[str, Any]:
        """
        Get fundamental data as dict (compatible with enhanced_predictor).
        """
        data = self.get_ticker(ticker)
        if not data:
            return {}
        
        return {
            'pe_ratio': data.pe_ratio,
            'peg_ratio': None,  # Not available from FinnHub
            'profit_margin': data.profit_margin,
            'revenue_growth': data.revenue_growth,
            'market_cap': data.market_cap,
            'analyst_score': data.analyst_score,
            'value_score': data.value_score,
        }


# Singleton instance
_client: Optional[FinnHubClient] = None


def get_finnhub_client() -> FinnHubClient:
    """Get singleton FinnHub client instance."""
    global _client
    if _client is None:
        _client = FinnHubClient()
    return _client


if __name__ == "__main__":
    # Test the client
    logging.basicConfig(level=logging.INFO)
    
    client = FinnHubClient()
    data = client.fetch_all()
    
    print(f"\nFetched {len(data)} tickers:")
    for ticker, info in data.items():
        print(f"\n{ticker}:")
        print(f"  Price: ${info.current_price:.2f}" if info.current_price else "  Price: N/A")
        print(f"  Change: {info.change_percent:.2f}%" if info.change_percent else "  Change: N/A")
        print(f"  P/E: {info.pe_ratio:.1f}" if info.pe_ratio else "  P/E: N/A")
        print(f"  Analyst Score: {info.analyst_score:.2f}")
        print(f"  Value Score: {info.value_score:.2f}")
