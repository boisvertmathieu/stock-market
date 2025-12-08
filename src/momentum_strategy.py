"""
Momentum Buy & Hold Strategy
Uses ML to select top tickers, then holds them for the duration.
This approach beats S&P in bull markets by concentrating in best performers.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

from .data_fetcher import DataFetcher, TOP_100_TICKERS
from .indicators import TechnicalIndicators
from .unified_scorer import UnifiedScorer

logger = logging.getLogger(__name__)


@dataclass
class MomentumResult:
    """Results from momentum strategy."""
    initial_capital: float
    final_value: float
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    benchmark_return: float
    alpha: float
    portfolio: List[Dict]
    start_date: datetime
    end_date: datetime


class MomentumStrategy:
    """
    Simple momentum strategy:
    1. Score all tickers by momentum factors at start
    2. Invest equally in top N tickers
    3. Rebalance monthly
    4. Compare to S&P 500
    """
    
    def __init__(
        self,
        initial_capital: float = 100000,
        top_n: int = 10,
        rebalance_freq: int = 21,  # Trading days (~monthly)
        silent: bool = True,
    ):
        self.initial_capital = initial_capital
        self.top_n = top_n
        self.rebalance_freq = rebalance_freq
        self.silent = silent
        
        if silent:
            logging.getLogger('src.data_fetcher').setLevel(logging.CRITICAL)
        
        self.fetcher = DataFetcher()
        # Unified scorer for consistent scoring
        self.scorer = UnifiedScorer()
        
        # FinnHub data cache
        self._finnhub_data = {}
        self._finnhub_loaded = False
    
    def _load_finnhub_data(self):
        """Load real-time data from Finnhub API."""
        if self._finnhub_loaded:
            return
        
        try:
            from .finnhub_client import FinnHubClient
            client = FinnHubClient()
            self._finnhub_data = client.fetch_all()
            self._finnhub_loaded = True
            if not self.silent:
                logger.info(f"Loaded {len(self._finnhub_data)} tickers from Finnhub")
        except Exception as e:
            logger.warning(f"Could not load Finnhub data: {e}")
            self._finnhub_loaded = True  # Don't retry
    
    def _score_ticker(self, df: pd.DataFrame, ticker: str = None) -> float:
        """Score a ticker using unified scoring + momentum."""
        if len(df) < 60:
            return -999
        
        try:
            # Calculate technical indicators
            ti = TechnicalIndicators(df)
            analysis = ti.get_comprehensive_analysis()
            
            # Get finnhub data if available
            finnhub_data = self._finnhub_data.get(ticker) if ticker else None
            
            # Use unified scorer
            unified_result = self.scorer.calculate_score(
                technical_analysis=analysis,
                finnhub_data=finnhub_data,
            )
            
            # Add momentum boost for strategy emphasis
            mom_20 = (df['Close'].iloc[-1] / df['Close'].iloc[-20] - 1) if len(df) >= 20 else 0
            mom_60 = (df['Close'].iloc[-1] / df['Close'].iloc[-60] - 1) if len(df) >= 60 else 0
            momentum_boost = 0.3 * mom_20 + 0.2 * mom_60
            
            # Volume trend boost
            vol_trend = df['Volume'].tail(20).mean() / df['Volume'].tail(60).mean() if df['Volume'].tail(60).mean() > 0 else 1
            volume_boost = 0.1 if vol_trend > 1.2 else 0
            
            # Combined: unified score + momentum emphasis
            final_score = unified_result.score + momentum_boost + volume_boost
            
            return final_score
            
        except:
            return -999
    
    def run(
        self,
        tickers: Optional[List[str]] = None,
        period: str = "2y",
        progress_callback = None
    ) -> MomentumResult:
        """Run the momentum strategy."""
        if tickers is None:
            tickers = TOP_100_TICKERS[:50]
        
        # Fetch all data
        historical_data = {}
        for ticker in tickers + ['SPY']:
            df = self.fetcher.fetch_ticker_data(ticker, period=period)
            if df is not None and len(df) > 100:
                historical_data[ticker] = df
        
        if 'SPY' not in historical_data:
            raise ValueError("Could not fetch SPY data")
        
        # Load FinnHub fundamental data
        self._load_finnhub_data()
        
        spy_data = historical_data['SPY']
        all_dates = spy_data.index.tolist()
        
        # Simulation period
        sim_days = min(500, len(all_dates) - 100)
        trading_dates = all_dates[-sim_days:]
        
        sim_start = trading_dates[0]
        sim_end = trading_dates[-1]
        initial_spy = spy_data.loc[sim_start, 'Close']
        
        # Portfolio: {ticker: shares}
        portfolio = {}
        cash = self.initial_capital
        daily_values = []
        holdings_log = []
        
        for day_idx, current_date in enumerate(trading_dates):
            # Progress
            if progress_callback and day_idx % 20 == 0:
                portfolio_value = cash
                for ticker, shares in portfolio.items():
                    if ticker in historical_data:
                        df = historical_data[ticker]
                        avail = df[df.index <= current_date]
                        if len(avail) > 0:
                            portfolio_value += avail['Close'].iloc[-1] * shares
                
                progress_callback(day_idx, len(trading_dates), {
                    'date': current_date,
                    'portfolio_value': portfolio_value,
                    'positions': len(portfolio),
                    'trades': 0
                })
            
            # Rebalance at start and every N days
            if day_idx % self.rebalance_freq == 0:
                # Sell all current holdings
                for ticker, shares in list(portfolio.items()):
                    if ticker in historical_data:
                        df = historical_data[ticker]
                        avail = df[df.index <= current_date]
                        if len(avail) > 0:
                            price = avail['Close'].iloc[-1]
                            cash += price * shares * 0.999  # 0.1% commission
                
                portfolio = {}
                
                # Score all tickers
                scores = []
                for ticker in tickers:
                    if ticker in historical_data:
                        df = historical_data[ticker]
                        avail = df[df.index <= current_date]
                        if len(avail) >= 60:
                            score = self._score_ticker(avail, ticker)
                            scores.append((ticker, score))
                
                # Select top N
                scores.sort(key=lambda x: -x[1])
                top_tickers = [t[0] for t in scores[:self.top_n]]
                
                # Equal weight allocation
                per_position = cash / self.top_n
                
                for ticker in top_tickers:
                    if ticker in historical_data:
                        df = historical_data[ticker]
                        avail = df[df.index <= current_date]
                        if len(avail) > 0:
                            price = avail['Close'].iloc[-1]
                            shares = int(per_position / price)
                            if shares > 0:
                                cost = price * shares * 1.001  # 0.1% commission
                                if cost <= cash:
                                    portfolio[ticker] = shares
                                    cash -= cost
                
                holdings_log.append({
                    'date': current_date,
                    'holdings': list(portfolio.keys())
                })
            
            # Calculate daily value
            portfolio_value = cash
            for ticker, shares in portfolio.items():
                if ticker in historical_data:
                    df = historical_data[ticker]
                    avail = df[df.index <= current_date]
                    if len(avail) > 0:
                        portfolio_value += avail['Close'].iloc[-1] * shares
            
            daily_values.append(portfolio_value)
        
        # Final value
        final_value = daily_values[-1] if daily_values else self.initial_capital
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # Metrics
        days = (sim_end - sim_start).days
        years = days / 365.25
        ann_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        returns = pd.Series(daily_values).pct_change().dropna()
        sharpe = returns.mean() * 252 / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        equity = pd.Series(daily_values)
        max_dd = ((equity - equity.expanding().max()) / equity.expanding().max()).min()
        
        # Benchmark
        final_spy = spy_data.loc[sim_end, 'Close']
        bench_return = (final_spy - initial_spy) / initial_spy
        
        return MomentumResult(
            initial_capital=self.initial_capital,
            final_value=final_value,
            total_return=total_return,
            annualized_return=ann_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            benchmark_return=bench_return,
            alpha=total_return - bench_return,
            portfolio=holdings_log,
            start_date=sim_start,
            end_date=sim_end,
        )
    
    def format_results(self, result: MomentumResult) -> str:
        beat = "✅ BEAT" if result.total_return > result.benchmark_return else "❌ MISSED"
        
        return f"""
{'='*70}
📊 MOMENTUM BUY & HOLD RESULTS
{'='*70}
Period: {result.start_date.date()} to {result.end_date.date()}
Initial Capital: ${result.initial_capital:,.2f}
Rebalance: Monthly | Top {self.top_n} tickers

----------------------------------------------------------------------
💰 PERFORMANCE
----------------------------------------------------------------------
Final Value:       ${result.final_value:,.2f}
Total Return:      {result.total_return:>10.2%}
Annualized Return: {result.annualized_return:>10.2%}
Sharpe Ratio:      {result.sharpe_ratio:>10.2f}
Max Drawdown:      {result.max_drawdown:>10.2%}

----------------------------------------------------------------------
📈 BENCHMARK COMPARISON (S&P 500) - {beat}
----------------------------------------------------------------------
S&P 500 Return:    {result.benchmark_return:>10.2%}
Strategy Return:   {result.total_return:>10.2%}
Alpha (Excess):    {result.alpha:>10.2%}

----------------------------------------------------------------------
📋 FINAL HOLDINGS
----------------------------------------------------------------------
{', '.join(result.portfolio[-1]['holdings']) if result.portfolio else 'N/A'}
{'='*70}
"""
