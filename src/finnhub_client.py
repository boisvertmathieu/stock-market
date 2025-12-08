"""
FinnHub Data Client - Direct API Access
Fetches real-time quotes, fundamentals, analyst ratings, and news directly from Finnhub API.
"""

import os
import requests
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

# Finnhub API configuration
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
FINNHUB_BASE_URL = "https://finnhub.io/api/v1"


@dataclass
class TickerData:
    """Container for ticker data from Finnhub API."""
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
    
    # News items for sentiment analysis
    news: List[Dict] = None
    
    # Analyst ratings
    analyst_buy: int = 0
    analyst_hold: int = 0
    analyst_sell: int = 0
    analyst_strong_buy: int = 0
    analyst_strong_sell: int = 0
    
    def __post_init__(self):
        if self.news is None:
            self.news = []
    
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
    
    @property
    def sentiment_score(self) -> float:
        """Calculate sentiment score from news (-1 to +1)."""
        if not self.news:
            return 0.0
        
        try:
            from .sentiment_analyzer import get_sentiment_analyzer
            analyzer = get_sentiment_analyzer()
            result = analyzer.analyze_news(self.news)
            return result.score
        except Exception as e:
            logger.warning(f"Failed to analyze sentiment for {self.ticker}: {e}")
            return 0.0
    
    @property
    def sentiment_confidence(self) -> float:
        """Get sentiment confidence (0-1)."""
        if not self.news:
            return 0.0
        
        try:
            from .sentiment_analyzer import get_sentiment_analyzer
            analyzer = get_sentiment_analyzer()
            result = analyzer.analyze_news(self.news)
            return result.confidence
        except Exception:
            return 0.0


class FinnHubClient:
    """Client to fetch data directly from Finnhub API."""
    
    def __init__(self, api_key: str = FINNHUB_API_KEY):
        self.api_key = api_key
        self.base_url = FINNHUB_BASE_URL
        self._cache: Dict[str, TickerData] = {}
        self._last_fetch = None
        self.last_error: Optional[str] = None
        self.last_fetch_success: bool = False
        self.timeout = int(os.getenv("FINNHUB_TIMEOUT", "30"))
        # Rate limiting: Finnhub free tier allows 60 calls/minute
        self.rate_limit_delay = float(os.getenv("FINNHUB_RATE_LIMIT_DELAY", "0.5"))
    
    @property
    def is_available(self) -> bool:
        """Check if client is properly configured and last fetch was successful."""
        return bool(self.api_key) and self.last_fetch_success and len(self._cache) > 0
    
    def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Optional[Dict]:
        """Make a request to Finnhub API with error handling."""
        if not self.api_key:
            logger.error("FINNHUB_API_KEY not configured")
            return None
        
        url = f"{self.base_url}{endpoint}"
        request_params = {"token": self.api_key}
        if params:
            request_params.update(params)
        
        try:
            response = requests.get(url, params=request_params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.warning(f"Finnhub API request failed for {endpoint}: {e}")
            return None
    
    def _fetch_quote(self, ticker: str) -> Dict[str, Any]:
        """Fetch real-time quote for a ticker."""
        data = self._make_request("/quote", {"symbol": ticker})
        if not data:
            return {}
        
        return {
            "current_price": data.get("c"),
            "change": data.get("d"),
            "change_percent": data.get("dp"),
            "high": data.get("h"),
            "low": data.get("l"),
            "open": data.get("o"),
            "previous_close": data.get("pc"),
        }
    
    def _fetch_metrics(self, ticker: str) -> Dict[str, Any]:
        """Fetch fundamental metrics for a ticker."""
        data = self._make_request("/stock/metric", {"symbol": ticker, "metric": "all"})
        if not data:
            return {}
        
        metric = data.get("metric", {})
        return {
            "pe_ratio": metric.get("peBasicExclExtraTTM"),
            "pe_forward": metric.get("peTTM"),
            "pb_ratio": metric.get("pbQuarterly"),
            "ps_ratio": metric.get("psTTM"),
            "eps": metric.get("epsBasicExclExtraItemsTTM"),
            "eps_growth": metric.get("epsGrowth5Y"),
            "revenue_growth": metric.get("revenueGrowth5Y"),
            "profit_margin": metric.get("netProfitMarginTTM"),
            "gross_margin": metric.get("grossMarginTTM"),
            "roe": metric.get("roeTTM"),
            "roa": metric.get("roaTTM"),
            "debt_equity": metric.get("totalDebtToEquityQuarterly"),
            "current_ratio": metric.get("currentRatioQuarterly"),
            "dividend_yield": metric.get("dividendYieldIndicatedAnnual"),
            "beta": metric.get("beta"),
            "market_cap": metric.get("marketCapitalization"),
            "week52_high": metric.get("52WeekHigh"),
            "week52_low": metric.get("52WeekLow"),
        }
    
    def _fetch_recommendations(self, ticker: str) -> Dict[str, Any]:
        """Fetch analyst recommendations for a ticker."""
        data = self._make_request("/stock/recommendation", {"symbol": ticker})
        if not data or not isinstance(data, list) or len(data) == 0:
            return {}
        
        # Get the latest recommendation
        latest = data[0]
        return {
            "analyst_buy": latest.get("buy", 0),
            "analyst_hold": latest.get("hold", 0),
            "analyst_sell": latest.get("sell", 0),
            "analyst_strong_buy": latest.get("strongBuy", 0),
            "analyst_strong_sell": latest.get("strongSell", 0),
        }
    
    def _fetch_news(self, ticker: str, days: int = 3) -> List[Dict]:
        """Fetch company news for a ticker."""
        today = datetime.now()
        from_date = (today - timedelta(days=days)).strftime("%Y-%m-%d")
        to_date = today.strftime("%Y-%m-%d")
        
        data = self._make_request("/company-news", {
            "symbol": ticker,
            "from": from_date,
            "to": to_date,
        })
        
        if not data or not isinstance(data, list):
            return []
        
        # Limit to 3 most recent news items (reduced for memory optimization)
        news = []
        for item in data[:3]:
            if item.get("headline"):
                news.append({
                    "headline": item.get("headline", ""),
                    "summary": item.get("summary", ""),
                    "datetime": item.get("datetime"),
                    "source": item.get("source", ""),
                    "url": item.get("url", ""),
                })
        return news
    
    def _fetch_ticker_data(self, ticker: str) -> Optional[TickerData]:
        """Fetch all data for a single ticker."""
        try:
            # Fetch all data types with rate limiting between calls
            quote = self._fetch_quote(ticker)
            time.sleep(self.rate_limit_delay)
            
            metrics = self._fetch_metrics(ticker)
            time.sleep(self.rate_limit_delay)
            
            recommendations = self._fetch_recommendations(ticker)
            time.sleep(self.rate_limit_delay)
            
            news = self._fetch_news(ticker)
            
            return TickerData(
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
                news=news,
                analyst_buy=recommendations.get("analyst_buy", 0),
                analyst_hold=recommendations.get("analyst_hold", 0),
                analyst_sell=recommendations.get("analyst_sell", 0),
                analyst_strong_buy=recommendations.get("analyst_strong_buy", 0),
                analyst_strong_sell=recommendations.get("analyst_strong_sell", 0),
            )
        except Exception as e:
            logger.warning(f"Failed to fetch data for {ticker}: {e}")
            return None
    
    def fetch_all(self, tickers: List[str] = None) -> Dict[str, TickerData]:
        """
        Fetch all ticker data from Finnhub API.
        
        Args:
            tickers: List of ticker symbols to fetch. If None, uses default list.
        
        Returns:
            Dictionary mapping ticker symbols to TickerData objects
        """
        if not self.api_key:
            logger.error("⚠️ FINNHUB_API_KEY not configured - cannot fetch data")
            self.last_error = "FINNHUB_API_KEY not configured"
            self.last_fetch_success = False
            return self._cache
        
        if tickers is None:
            # Default tickers if none provided
            tickers = ["AAPL", "META", "NVDA", "TSLA", "MSFT", "AMZN", 
                       "GOOGL", "AMD", "NFLX", "JPM", "SPY", "QQQ"]
        
        try:
            result = {}
            total = len(tickers)
            
            for i, ticker in enumerate(tickers, 1):
                logger.info(f"Fetching {ticker} ({i}/{total})...")
                ticker_data = self._fetch_ticker_data(ticker)
                if ticker_data:
                    result[ticker] = ticker_data
                
                # Add delay between tickers to respect rate limits
                if i < total:
                    time.sleep(self.rate_limit_delay)
            
            self._cache = result
            self._last_fetch = datetime.now()
            self.last_fetch_success = True
            self.last_error = None
            logger.info(f"✅ Fetched {len(result)} tickers from Finnhub API")
            return result
            
        except Exception as e:
            self.last_error = str(e)
            self.last_fetch_success = False
            logger.warning(f"⚠️ Finnhub API error: {e}")
            if self._cache:
                logger.warning(f"Using cached data ({len(self._cache)} tickers) - may be stale")
            else:
                logger.warning("No cached data available - strategies will use technical-only scoring")
            return self._cache
    
    def get_ticker(self, ticker: str) -> Optional[TickerData]:
        """Get data for a specific ticker."""
        if ticker in self._cache:
            return self._cache[ticker]
        
        # Fetch just this ticker if not cached
        ticker_data = self._fetch_ticker_data(ticker)
        if ticker_data:
            self._cache[ticker] = ticker_data
        return ticker_data
    
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
