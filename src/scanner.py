"""
Market Scanner Module
Parallel scanning of multiple tickers with signal aggregation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import logging
import json
from datetime import datetime

from .data_fetcher import DataFetcher, TOP_100_TICKERS
from .indicators import TechnicalIndicators, Signal
from .predictor import MLPredictor, PredictionClass
from .unified_scorer import UnifiedScorer, SignalStrength
from .finnhub_client import get_finnhub_client

logger = logging.getLogger(__name__)


@dataclass
class TickerAnalysis:
    """Complete analysis for a single ticker."""
    ticker: str
    name: str
    sector: str
    price: float
    change: float
    change_percent: float
    volume: int
    avg_volume: int
    volume_ratio: float
    technical_signal: str
    technical_score: float
    ml_prediction: str
    ml_confidence: str
    ml_probability: float
    overall_action: str
    action_strength: int  # -2 to 2
    key_factors: List[str]
    entry_price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    risk_reward_ratio: Optional[float]
    timestamp: str


class MarketScanner:
    """
    Scans multiple tickers in parallel and generates trading recommendations.
    """
    
    def __init__(self, max_workers: int = 10):
        """
        Initialize scanner.
        
        Args:
            max_workers: Maximum parallel workers for scanning
        """
        self.max_workers = max_workers
        self.fetcher = DataFetcher()
        self.results: Dict[str, TickerAnalysis] = {}
        self.scorer = UnifiedScorer()
        self.finnhub = get_finnhub_client()
        
    def _analyze_single_ticker(self, ticker: str, period: str = "1y") -> Optional[TickerAnalysis]:
        """
        Analyze a single ticker.
        
        Args:
            ticker: Stock ticker symbol
            period: Data period to fetch
            
        Returns:
            TickerAnalysis or None if analysis failed
        """
        try:
            # Fetch data
            df = self.fetcher.fetch_ticker_data(ticker, period=period)
            if df is None or len(df) < 50:
                return None
            
            # Get ticker info
            info = self.fetcher.get_ticker_info(ticker)
            
            # Calculate indicators
            ti = TechnicalIndicators(df)
            analysis = ti.get_comprehensive_analysis()
            
            # ML Prediction
            predictor = MLPredictor()
            df_with_indicators = ti.get_dataframe()
            predictor.train(df_with_indicators, n_splits=3)
            ml_result = predictor.predict(df_with_indicators)
            
            # Use unified scorer for consistent scoring
            finnhub_data = self.finnhub._cache.get(ticker)
            unified_result = self.scorer.calculate_score(
                technical_analysis=analysis,
                ml_result=ml_result,
                finnhub_data=finnhub_data,
            )
            
            combined_score = unified_result.score
            action = self.scorer.get_signal_name(unified_result).upper()
            action_strength = unified_result.signal.value
            
            # Extract key factors
            key_factors = []
            for name, ind in analysis['indicators'].items():
                if abs(ind['signal_value']) >= 1:
                    key_factors.append(ind['description'])
            key_factors = key_factors[:3]  # Top 3 factors
            
            # Calculate trade levels
            current_price = analysis['price']['current']
            atr = analysis['raw_indicators'].get('ATR', current_price * 0.02)
            
            if action_strength > 0:  # Buy
                entry = current_price
                stop_loss = current_price - 2 * atr
                take_profit = current_price + 3 * atr
            elif action_strength < 0:  # Sell/Short
                entry = current_price
                stop_loss = current_price + 2 * atr
                take_profit = current_price - 3 * atr
            else:
                entry = None
                stop_loss = None
                take_profit = None
            
            # Risk/Reward calculation
            if entry and stop_loss and take_profit:
                risk = abs(entry - stop_loss)
                reward = abs(take_profit - entry)
                rr_ratio = reward / risk if risk > 0 else 0
            else:
                rr_ratio = None
            
            # Build result
            return TickerAnalysis(
                ticker=ticker,
                name=info.get('name', ticker),
                sector=info.get('sector', 'N/A'),
                price=current_price,
                change=analysis['price']['change'],
                change_percent=analysis['price']['change_percent'],
                volume=int(df['Volume'].iloc[-1]),
                avg_volume=int(info.get('avg_volume', 0)),
                volume_ratio=round(analysis['raw_indicators'].get('Volume_Ratio', 1.0), 2),
                technical_signal=analysis['overall']['signal'],
                technical_score=analysis['overall']['score'],
                ml_prediction=ml_result.prediction.name if ml_result else "N/A",
                ml_confidence=ml_result.confidence if ml_result else "LOW",
                ml_probability=ml_result.probability if ml_result else 0,
                overall_action=action,
                action_strength=action_strength,
                key_factors=key_factors,
                entry_price=round(entry, 2) if entry else None,
                stop_loss=round(stop_loss, 2) if stop_loss else None,
                take_profit=round(take_profit, 2) if take_profit else None,
                risk_reward_ratio=round(rr_ratio, 2) if rr_ratio else None,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"Error analyzing {ticker}: {e}")
            return None
    
    def scan_tickers(
        self,
        tickers: Optional[List[str]] = None,
        period: str = "1y",
        progress_callback: callable = None
    ) -> List[TickerAnalysis]:
        """
        Scan multiple tickers in parallel.
        
        Args:
            tickers: List of tickers to scan (defaults to TOP_100)
            period: Data period
            progress_callback: Optional callback(completed, total, ticker)
            
        Returns:
            List of TickerAnalysis sorted by action strength
        """
        if tickers is None:
            tickers = TOP_100_TICKERS
        
        results = []
        completed = 0
        total = len(tickers)
        
        # Use ThreadPoolExecutor for I/O-bound operations
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_ticker = {
                executor.submit(self._analyze_single_ticker, ticker, period): ticker
                for ticker in tickers
            }
            
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        self.results[ticker] = result
                except Exception as e:
                    logger.error(f"Error processing {ticker}: {e}")
                
                completed += 1
                if progress_callback:
                    progress_callback(completed, total, ticker)
        
        # Sort by action strength (strongest signals first)
        results.sort(key=lambda x: (-abs(x.action_strength), -x.ml_probability))
        
        return results
    
    def get_top_opportunities(
        self,
        results: Optional[List[TickerAnalysis]] = None,
        action_type: str = "BUY",
        limit: int = 10
    ) -> List[TickerAnalysis]:
        """
        Get top opportunities filtered by action type.
        
        Args:
            results: List of analyses (defaults to cached results)
            action_type: "BUY", "SELL", or "ALL"
            limit: Maximum results to return
            
        Returns:
            Filtered and sorted list
        """
        if results is None:
            results = list(self.results.values())
        
        if action_type == "BUY":
            filtered = [r for r in results if r.action_strength > 0]
            filtered.sort(key=lambda x: (-x.action_strength, -x.ml_probability))
        elif action_type == "SELL":
            filtered = [r for r in results if r.action_strength < 0]
            filtered.sort(key=lambda x: (x.action_strength, -x.ml_probability))
        else:
            filtered = [r for r in results if r.action_strength != 0]
            filtered.sort(key=lambda x: (-abs(x.action_strength), -x.ml_probability))
        
        return filtered[:limit]
    
    def to_json(self, results: List[TickerAnalysis]) -> str:
        """Convert results to JSON for TUI consumption."""
        return json.dumps([asdict(r) for r in results], indent=2)
    
    def get_market_summary(self, results: List[TickerAnalysis]) -> Dict[str, Any]:
        """Generate market summary from scan results."""
        if not results:
            return {"error": "No results"}
        
        bullish = len([r for r in results if r.action_strength > 0])
        bearish = len([r for r in results if r.action_strength < 0])
        neutral = len([r for r in results if r.action_strength == 0])
        
        avg_change = np.mean([r.change_percent for r in results])
        
        # Group by sector
        sector_sentiment = {}
        for r in results:
            if r.sector not in sector_sentiment:
                sector_sentiment[r.sector] = []
            sector_sentiment[r.sector].append(r.action_strength)
        
        sector_summary = {
            sector: {"avg_signal": np.mean(signals), "count": len(signals)}
            for sector, signals in sector_sentiment.items()
        }
        
        # Top movers
        top_gainers = sorted(results, key=lambda x: -x.change_percent)[:5]
        top_losers = sorted(results, key=lambda x: x.change_percent)[:5]
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_scanned": len(results),
            "market_sentiment": {
                "bullish": bullish,
                "bearish": bearish,
                "neutral": neutral,
                "ratio": bullish / bearish if bearish > 0 else float('inf')
            },
            "avg_change_percent": round(avg_change, 2),
            "sector_summary": sector_summary,
            "top_gainers": [{"ticker": r.ticker, "change": r.change_percent} for r in top_gainers],
            "top_losers": [{"ticker": r.ticker, "change": r.change_percent} for r in top_losers],
            "strong_buy_count": len([r for r in results if r.action_strength == 2]),
            "strong_sell_count": len([r for r in results if r.action_strength == -2]),
        }
    
    def format_results_table(self, results: List[TickerAnalysis], limit: int = 20) -> str:
        """Format results as ASCII table."""
        lines = [
            "=" * 100,
            f"{'TICKER':<8} {'NAME':<20} {'PRICE':>10} {'CHG%':>8} {'SIGNAL':>12} {'ML':>10} {'ACTION':>15}",
            "=" * 100,
        ]
        
        for r in results[:limit]:
            name = r.name[:18] + ".." if len(r.name) > 20 else r.name
            chg = f"{r.change_percent:+.2f}%"
            lines.append(
                f"{r.ticker:<8} {name:<20} ${r.price:>9.2f} {chg:>8} "
                f"{r.technical_signal:>12} {r.ml_prediction:>10} {r.overall_action:>15}"
            )
        
        lines.append("=" * 100)
        return "\n".join(lines)


if __name__ == "__main__":
    import sys
    
    print("Starting market scan...")
    scanner = MarketScanner(max_workers=5)
    
    # Test with small subset
    test_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    
    def progress(completed, total, ticker):
        print(f"Progress: {completed}/{total} - {ticker}")
    
    results = scanner.scan_tickers(tickers=test_tickers, progress_callback=progress)
    
    print("\n" + scanner.format_results_table(results))
    
    print("\nTop Buy Opportunities:")
    for r in scanner.get_top_opportunities(results, "BUY", 3):
        print(f"  {r.ticker}: {r.overall_action} - {', '.join(r.key_factors)}")
    
    print("\nMarket Summary:")
    summary = scanner.get_market_summary(results)
    print(json.dumps(summary, indent=2))
