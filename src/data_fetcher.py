"""
Data Fetcher Module
Handles fetching stock data from Yahoo Finance with caching and rate limit handling.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Top 100 Most Traded US Stocks (by volume and market cap)
TOP_100_TICKERS = [
    # FAANG + Major Tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "INTC", "CRM",
    "ORCL", "ADBE", "NFLX", "PYPL", "UBER", "SQ", "SHOP", "SNOW", "PLTR", "COIN",
    # Finance
    "JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "V", "MA", "BLK",
    # Healthcare & Pharma
    "JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY", "TMO", "ABT", "BMY", "GILD",
    # Consumer
    "WMT", "COST", "HD", "TGT", "NKE", "SBUX", "MCD", "KO", "PEP", "PG",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "OXY", "MPC", "VLO", "PSX", "HAL",
    # Industrial & Transport
    "CAT", "DE", "BA", "RTX", "LMT", "UPS", "FDX", "DAL", "UAL", "AAL",
    # Semiconductors
    "AVGO", "QCOM", "TXN", "MU", "LRCX", "AMAT", "KLAC", "MRVL", "ON", "SWKS",
    # Telecom & Media
    "T", "VZ", "TMUS", "DIS", "CMCSA", "CHTR", "WBD", "PARA", "FOX", "NWSA",
    # ETFs (Popular)
    "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "ARKK", "XLF", "XLE", "XLK"
]


class DataFetcher:
    """
    Fetches and caches stock market data from Yahoo Finance.
    Implements rate limiting and retry logic for robustness.
    """
    
    def __init__(self, cache_hours: int = 1):
        """
        Initialize the DataFetcher.
        
        Args:
            cache_hours: Number of hours to cache data before refreshing
        """
        self.cache_hours = cache_hours
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._last_fetch: Dict[str, datetime] = {}
        self.max_retries = 3
        self.retry_delay = 1.0  # seconds
        
    def _is_cache_valid(self, ticker: str) -> bool:
        """Check if cached data is still valid."""
        if ticker not in self._cache or ticker not in self._last_fetch:
            return False
        age = datetime.now() - self._last_fetch[ticker]
        return age < timedelta(hours=self.cache_hours)
    
    def fetch_ticker_data(
        self,
        ticker: str,
        period: str = "2y",
        interval: str = "1d",
        force_refresh: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical data for a single ticker.
        
        Args:
            ticker: Stock ticker symbol
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max)
            interval: Data interval (1m, 5m, 15m, 1h, 1d, 1wk, 1mo)
            force_refresh: Force fetch even if cache is valid
            
        Returns:
            DataFrame with OHLCV data or None if fetch failed
        """
        cache_key = f"{ticker}_{period}_{interval}"
        
        if not force_refresh and self._is_cache_valid(cache_key):
            return self._cache[cache_key]['data']
        
        for attempt in range(self.max_retries):
            try:
                stock = yf.Ticker(ticker)
                df = stock.history(period=period, interval=interval)
                
                if df.empty:
                    logger.warning(f"No data returned for {ticker}")
                    return None
                
                # Clean data
                df = df.dropna()
                df.index = pd.to_datetime(df.index)
                
                # Cache the result
                self._cache[cache_key] = {
                    'data': df,
                    'info': self._get_ticker_info(stock)
                }
                self._last_fetch[cache_key] = datetime.now()
                
                return df
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{self.max_retries} failed for {ticker}: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (attempt + 1))
                    
        logger.error(f"Failed to fetch data for {ticker} after {self.max_retries} attempts")
        return None
    
    def _get_ticker_info(self, stock: yf.Ticker) -> Dict[str, Any]:
        """Extract relevant info from ticker object."""
        try:
            info = stock.info
            return {
                'name': info.get('shortName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', None),
                'eps': info.get('trailingEps', None),
                'dividend_yield': info.get('dividendYield', None),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh', None),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow', None),
                'avg_volume': info.get('averageVolume', 0),
            }
        except Exception:
            return {}
    
    def get_ticker_info(self, ticker: str) -> Dict[str, Any]:
        """Get cached ticker info or fetch it."""
        cache_key = f"{ticker}_2y_1d"
        if cache_key in self._cache and 'info' in self._cache[cache_key]:
            return self._cache[cache_key]['info']
        
        # Fetch to populate cache
        self.fetch_ticker_data(ticker)
        if cache_key in self._cache:
            return self._cache[cache_key].get('info', {})
        return {}
    
    def fetch_multiple_tickers(
        self,
        tickers: List[str],
        period: str = "2y",
        interval: str = "1d",
        max_workers: int = 10,
        progress_callback: callable = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple tickers in parallel.
        
        Args:
            tickers: List of ticker symbols
            period: Data period
            interval: Data interval
            max_workers: Maximum parallel workers
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary mapping tickers to their DataFrames
        """
        results = {}
        completed = 0
        total = len(tickers)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {
                executor.submit(self.fetch_ticker_data, ticker, period, interval): ticker
                for ticker in tickers
            }
            
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    df = future.result()
                    if df is not None:
                        results[ticker] = df
                except Exception as e:
                    logger.error(f"Error fetching {ticker}: {e}")
                
                completed += 1
                if progress_callback:
                    progress_callback(completed, total, ticker)
                    
        return results
    
    def get_top_100_tickers(self) -> List[str]:
        """Return the list of top 100 most traded tickers."""
        return TOP_100_TICKERS.copy()
    
    def get_current_price(self, ticker: str) -> Optional[Dict[str, float]]:
        """
        Get current price info for a ticker.
        
        Returns:
            Dict with current price, change, change_percent
        """
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period="2d")
            
            if len(hist) < 1:
                return None
                
            current = hist['Close'].iloc[-1]
            previous = hist['Close'].iloc[-2] if len(hist) > 1 else current
            change = current - previous
            change_pct = (change / previous) * 100 if previous != 0 else 0
            
            return {
                'price': round(current, 2),
                'change': round(change, 2),
                'change_percent': round(change_pct, 2),
                'volume': int(hist['Volume'].iloc[-1]),
            }
        except Exception as e:
            logger.error(f"Error getting current price for {ticker}: {e}")
            return None
    
    def export_to_json(self, ticker: str) -> str:
        """Export ticker data to JSON for TUI consumption."""
        cache_key = f"{ticker}_2y_1d"
        if cache_key not in self._cache:
            self.fetch_ticker_data(ticker)
            
        if cache_key not in self._cache:
            return json.dumps({"error": f"No data for {ticker}"})
        
        data = self._cache[cache_key]
        df = data['data']
        
        # Convert DataFrame to JSON-serializable format
        return json.dumps({
            'ticker': ticker,
            'info': data.get('info', {}),
            'data': {
                'dates': df.index.strftime('%Y-%m-%d').tolist()[-60:],  # Last 60 days
                'open': df['Open'].round(2).tolist()[-60:],
                'high': df['High'].round(2).tolist()[-60:],
                'low': df['Low'].round(2).tolist()[-60:],
                'close': df['Close'].round(2).tolist()[-60:],
                'volume': df['Volume'].astype(int).tolist()[-60:],
            }
        })


if __name__ == "__main__":
    # Quick test
    fetcher = DataFetcher()
    print(f"Testing with {fetcher.get_top_100_tickers()[:5]}...")
    
    data = fetcher.fetch_ticker_data("AAPL")
    if data is not None:
        print(f"AAPL: {len(data)} rows fetched")
        print(data.tail())
    
    info = fetcher.get_ticker_info("AAPL")
    print(f"Info: {info}")
