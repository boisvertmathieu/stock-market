"""
Trading Simulator Module
Simulates daily trading strategy over historical data to evaluate performance vs S&P 500.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import logging
import json

from .data_fetcher import DataFetcher, TOP_100_TICKERS
from .indicators import TechnicalIndicators, Signal
from .predictor import MLPredictor, PredictionClass

logger = logging.getLogger(__name__)


class TradeDirection(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


@dataclass
class Trade:
    """Represents a completed trade."""
    ticker: str
    direction: TradeDirection
    entry_date: datetime
    entry_price: float
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    shares: int = 0
    pnl: float = 0.0
    pnl_percent: float = 0.0
    exit_reason: str = ""
    signal_score: float = 0.0
    
    @property
    def is_open(self) -> bool:
        return self.exit_date is None


@dataclass
class Position:
    """Represents an open position."""
    ticker: str
    direction: TradeDirection
    entry_date: datetime
    entry_price: float
    shares: int
    stop_loss: float
    take_profit: float
    signal_score: float
    
    def calculate_pnl(self, current_price: float) -> float:
        if self.direction == TradeDirection.LONG:
            return (current_price - self.entry_price) * self.shares
        else:
            return (self.entry_price - current_price) * self.shares
    
    def calculate_pnl_percent(self, current_price: float) -> float:
        if self.direction == TradeDirection.LONG:
            return (current_price - self.entry_price) / self.entry_price * 100
        else:
            return (self.entry_price - current_price) / self.entry_price * 100


@dataclass
class DailySnapshot:
    """Daily portfolio snapshot."""
    date: datetime
    portfolio_value: float
    cash: float
    positions_value: float
    num_positions: int
    daily_return: float
    cumulative_return: float
    benchmark_value: float
    benchmark_return: float
    trades_today: int


@dataclass
class SimulationResult:
    """Complete simulation results."""
    # Configuration
    initial_capital: float
    start_date: datetime
    end_date: datetime
    
    # Performance metrics
    final_value: float
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    
    # Benchmark comparison
    benchmark_return: float
    alpha: float  # Excess return vs benchmark
    
    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_trade_pnl: float
    avg_win: float
    avg_loss: float
    best_trade: float
    worst_trade: float
    
    # Data
    trades: List[Trade] = field(default_factory=list)
    daily_snapshots: List[DailySnapshot] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    benchmark_curve: pd.Series = field(default_factory=pd.Series)


class TradingSimulator:
    """
    Simulates a daily LONG-ONLY trading strategy over historical data.
    Uses technical indicators and ML predictions to select trades.
    
    Note: This is a long-only strategy. Short selling is not implemented.
    """
    
    def __init__(
        self,
        initial_capital: float = 100000,
        max_positions: int = 5,
        position_size_pct: float = 0.20,
        commission: float = 0.001,
        slippage: float = 0.001,
        stop_loss_atr_mult: float = 2.0,
        take_profit_atr_mult: float = 3.0,
        silent: bool = False,
    ):
        """
        Initialize the simulator.
        
        Args:
            initial_capital: Starting capital
            max_positions: Maximum simultaneous positions
            position_size_pct: Position size as fraction of capital
            commission: Commission per trade
            slippage: Slippage per trade
            stop_loss_atr_mult: ATR multiplier for stop-loss
            take_profit_atr_mult: ATR multiplier for take-profit
            silent: If True, suppress all logging
        """
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.position_size_pct = position_size_pct
        self.commission = commission
        self.slippage = slippage
        self.stop_loss_atr_mult = stop_loss_atr_mult
        self.take_profit_atr_mult = take_profit_atr_mult
        self.silent = silent
        
        # Suppress logging if silent
        if silent:
            logging.getLogger('src.predictor').setLevel(logging.CRITICAL)
            logging.getLogger('src.indicators').setLevel(logging.CRITICAL)
            logging.getLogger('src.data_fetcher').setLevel(logging.CRITICAL)
        
        self.fetcher = DataFetcher()
        self.predictors: Dict[str, MLPredictor] = {}
        
        # State
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.daily_snapshots: List[DailySnapshot] = []
        
    def _get_predictor(self, ticker: str) -> MLPredictor:
        """Get or create predictor for ticker."""
        if ticker not in self.predictors:
            self.predictors[ticker] = MLPredictor()
        return self.predictors[ticker]
    
    def _analyze_ticker(
        self,
        ticker: str,
        historical_data: Dict[str, pd.DataFrame],
        current_date: datetime,
        lookback_days: int = 300
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze a ticker at a specific point in time.
        Uses only data available up to current_date.
        """
        if ticker not in historical_data:
            return None
        
        df = historical_data[ticker]
        
        # Filter to data available at current_date
        available_data = df[df.index <= current_date]
        
        # Need more data for proper indicator calculation
        if len(available_data) < 100:
            return None
        
        # Use last N days for analysis (need enough for rolling windows)
        analysis_data = available_data.tail(lookback_days)
        
        try:
            # Calculate indicators
            ti = TechnicalIndicators(analysis_data)
            analysis = ti.get_comprehensive_analysis()
            
            # Get current price and ATR first (for fallback)
            current_price = analysis_data['Close'].iloc[-1]
            atr = analysis['raw_indicators'].get('ATR', current_price * 0.02)
            
            # Calculate technical score
            tech_score = analysis['overall']['signal_value']
            
            # Get ML prediction only if predictor is available and trained
            predictor = self._get_predictor(ticker)
            ml_score = 0
            ml_pred_name = 'N/A'
            
            # Only use ML if we have enough data and model is trained
            if predictor.is_trained:
                try:
                    df_with_indicators = ti.get_dataframe()
                    ml_result = predictor.predict(df_with_indicators)
                    if ml_result:
                        ml_score = ml_result.prediction.value
                        ml_pred_name = ml_result.prediction.name
                except:
                    pass
            
            # Combined score (rely more on technical if ML not available)
            if ml_score != 0:
                combined_score = 0.5 * tech_score + 0.5 * ml_score
            else:
                combined_score = tech_score  # Use only technical
            
            return {
                'ticker': ticker,
                'date': current_date,
                'price': current_price,
                'technical_signal': analysis['overall']['signal'],
                'technical_score': tech_score,
                'ml_prediction': ml_pred_name,
                'ml_score': ml_score,
                'combined_score': combined_score,
                'atr': atr,
                'volume_ratio': analysis['raw_indicators'].get('Volume_Ratio', 1.0),
            }
        except Exception as e:
            if not self.silent:
                logger.debug(f"Error analyzing {ticker}: {e}")
            return None
    
    def _execute_trade(
        self,
        ticker: str,
        direction: TradeDirection,
        price: float,
        date: datetime,
        atr: float,
        signal_score: float
    ) -> bool:
        """Execute a new trade (open position)."""
        if len(self.positions) >= self.max_positions:
            return False
        
        if ticker in self.positions:
            return False
        
        # Calculate position size
        position_value = self.cash * self.position_size_pct
        if position_value <= 0:
            return False
        
        # Apply slippage
        if direction == TradeDirection.LONG:
            entry_price = price * (1 + self.slippage)
            stop_loss = entry_price - self.stop_loss_atr_mult * atr
            take_profit = entry_price + self.take_profit_atr_mult * atr
        else:
            entry_price = price * (1 - self.slippage)
            stop_loss = entry_price + self.stop_loss_atr_mult * atr
            take_profit = entry_price - self.take_profit_atr_mult * atr
        
        shares = int(position_value / entry_price)
        if shares <= 0:
            return False
        
        cost = shares * entry_price * (1 + self.commission)
        if cost > self.cash:
            shares = int((self.cash / (1 + self.commission)) / entry_price)
            cost = shares * entry_price * (1 + self.commission)
        
        if shares <= 0:
            return False
        
        self.cash -= cost
        
        self.positions[ticker] = Position(
            ticker=ticker,
            direction=direction,
            entry_date=date,
            entry_price=entry_price,
            shares=shares,
            stop_loss=stop_loss,
            take_profit=take_profit,
            signal_score=signal_score
        )
        
        return True
    
    def _close_position(
        self,
        ticker: str,
        price: float,
        date: datetime,
        reason: str
    ) -> Optional[Trade]:
        """Close an existing position."""
        if ticker not in self.positions:
            return None
        
        pos = self.positions[ticker]
        
        # Apply slippage
        if pos.direction == TradeDirection.LONG:
            exit_price = price * (1 - self.slippage)
        else:
            exit_price = price * (1 + self.slippage)
        
        # Calculate P&L
        if pos.direction == TradeDirection.LONG:
            pnl = (exit_price - pos.entry_price) * pos.shares
        else:
            pnl = (pos.entry_price - exit_price) * pos.shares
        
        # Subtract commission
        pnl -= exit_price * pos.shares * self.commission
        
        pnl_percent = pnl / (pos.entry_price * pos.shares) * 100
        
        # Add cash back
        self.cash += exit_price * pos.shares * (1 - self.commission)
        
        # Record trade
        trade = Trade(
            ticker=ticker,
            direction=pos.direction,
            entry_date=pos.entry_date,
            entry_price=pos.entry_price,
            exit_date=date,
            exit_price=exit_price,
            shares=pos.shares,
            pnl=pnl,
            pnl_percent=pnl_percent,
            exit_reason=reason,
            signal_score=pos.signal_score
        )
        
        self.trades.append(trade)
        del self.positions[ticker]
        
        return trade
    
    def _check_stop_loss_take_profit(
        self,
        ticker: str,
        current_price: float,
        date: datetime
    ) -> bool:
        """Check if stop-loss or take-profit is hit."""
        if ticker not in self.positions:
            return False
        
        pos = self.positions[ticker]
        
        if pos.direction == TradeDirection.LONG:
            if current_price <= pos.stop_loss:
                self._close_position(ticker, current_price, date, "STOP_LOSS")
                return True
            if current_price >= pos.take_profit:
                self._close_position(ticker, current_price, date, "TAKE_PROFIT")
                return True
        else:
            if current_price >= pos.stop_loss:
                self._close_position(ticker, current_price, date, "STOP_LOSS")
                return True
            if current_price <= pos.take_profit:
                self._close_position(ticker, current_price, date, "TAKE_PROFIT")
                return True
        
        return False
    
    def _calculate_portfolio_value(
        self,
        historical_data: Dict[str, pd.DataFrame],
        date: datetime
    ) -> float:
        """Calculate total portfolio value at a given date."""
        positions_value = 0.0
        
        for ticker, pos in self.positions.items():
            if ticker in historical_data:
                df = historical_data[ticker]
                available = df[df.index <= date]
                if len(available) > 0:
                    current_price = available['Close'].iloc[-1]
                    if pos.direction == TradeDirection.LONG:
                        positions_value += current_price * pos.shares
                    else:
                        # For short: value = entry_value + unrealized_pnl
                        unrealized_pnl = (pos.entry_price - current_price) * pos.shares
                        positions_value += pos.entry_price * pos.shares + unrealized_pnl
        
        return self.cash + positions_value
    
    def run(
        self,
        tickers: Optional[List[str]] = None,
        period: str = "2y",
        start_date: Optional[datetime] = None,
        progress_callback: callable = None
    ) -> SimulationResult:
        """
        Run the trading simulation.
        
        Args:
            tickers: List of tickers to trade (default: top 50)
            period: Historical data period
            start_date: Start date for simulation (default: 1 year before end)
            progress_callback: Optional progress callback(day, total_days, info)
            
        Returns:
            SimulationResult with all metrics and trade history
        """
        if tickers is None:
            tickers = TOP_100_TICKERS[:50]  # Use top 50 for speed
        
        logger.info(f"Starting simulation with {len(tickers)} tickers...")
        
        # Reset state
        self.cash = self.initial_capital
        self.positions = {}
        self.trades = []
        self.daily_snapshots = []
        self.predictors = {}
        
        # Fetch all historical data
        logger.info("Fetching historical data...")
        historical_data = {}
        for ticker in tickers + ['SPY']:  # Include SPY for benchmark
            df = self.fetcher.fetch_ticker_data(ticker, period=period)
            if df is not None and len(df) > 100:
                historical_data[ticker] = df
        
        if 'SPY' not in historical_data:
            raise ValueError("Could not fetch SPY data for benchmark")
        
        # Get trading dates from SPY
        spy_data = historical_data['SPY']
        all_dates = spy_data.index.tolist()
        
        # Determine simulation period (use last 500 trading days ~ 2 years)
        simulation_days = min(500, len(all_dates) - 100)  # Keep 100 days for initial training
        trading_dates = all_dates[-simulation_days:]
        
        sim_start_date = trading_dates[0]
        sim_end_date = trading_dates[-1]
        
        # Initial benchmark value
        initial_spy_price = spy_data.loc[sim_start_date, 'Close']
        
        logger.info(f"Simulating from {sim_start_date.date()} to {sim_end_date.date()}")
        logger.info(f"Trading {len(trading_dates)} days with {len(historical_data)} tickers")
        
        # Pre-train ML models with data before simulation start
        warmup_date = trading_dates[0]
        for ticker in tickers:
            if ticker in historical_data:
                df = historical_data[ticker]
                warmup_data = df[df.index < warmup_date]
                if len(warmup_data) >= 100:
                    try:
                        ti = TechnicalIndicators(warmup_data.tail(300))
                        df_indicators = ti.get_dataframe()
                        predictor = self._get_predictor(ticker)
                        predictor.train(df_indicators, n_splits=2)
                    except:
                        pass  # Skip tickers with insufficient data
        
        # Run simulation day by day
        prev_portfolio_value = self.initial_capital
        
        for day_idx, current_date in enumerate(trading_dates):
            trades_today = 0
            
            # Progress callback
            if progress_callback and day_idx % 10 == 0:
                progress_callback(day_idx, len(trading_dates), {
                    'date': current_date,
                    'portfolio_value': prev_portfolio_value,
                    'positions': len(self.positions),
                    'trades': len(self.trades)
                })
            
            # 1. Check stop-loss/take-profit for existing positions
            for ticker in list(self.positions.keys()):
                if ticker in historical_data:
                    df = historical_data[ticker]
                    available = df[df.index <= current_date]
                    if len(available) > 0:
                        current_price = available['Close'].iloc[-1]
                        if self._check_stop_loss_take_profit(ticker, current_price, current_date):
                            trades_today += 1
            
            # 2. Analyze all tickers
            ticker_analyses = []
            for ticker in tickers:
                if ticker in historical_data:
                    analysis = self._analyze_ticker(ticker, historical_data, current_date)
                    if analysis:
                        ticker_analyses.append(analysis)
            
            # 3. Sort by signal strength (LONG-only strategy)
            buy_candidates = [a for a in ticker_analyses if a['combined_score'] >= 1.0]
            buy_candidates.sort(key=lambda x: -x['combined_score'])
            
            # 4. Check if we should close LONG positions based on signals
            for ticker in list(self.positions.keys()):
                pos = self.positions[ticker]
                
                # Find current analysis for this ticker
                ticker_analysis = next((a for a in ticker_analyses if a['ticker'] == ticker), None)
                
                if ticker_analysis:
                    # Close LONG if signal turns bearish
                    if pos.direction == TradeDirection.LONG and ticker_analysis['combined_score'] <= -0.5:
                        if ticker in historical_data:
                            df = historical_data[ticker]
                            available = df[df.index <= current_date]
                            if len(available) > 0:
                                self._close_position(ticker, available['Close'].iloc[-1], current_date, "SIGNAL_REVERSE")
                                trades_today += 1
            
            # 5. Open new positions if we have room
            for candidate in buy_candidates[:3]:  # Top 3 buy signals
                if len(self.positions) < self.max_positions:
                    ticker = candidate['ticker']
                    if ticker not in self.positions:
                        if self._execute_trade(
                            ticker,
                            TradeDirection.LONG,
                            candidate['price'],
                            current_date,
                            candidate['atr'],
                            candidate['combined_score']
                        ):
                            trades_today += 1
            
            # 6. Calculate daily snapshot
            portfolio_value = self._calculate_portfolio_value(historical_data, current_date)
            daily_return = (portfolio_value - prev_portfolio_value) / prev_portfolio_value if prev_portfolio_value > 0 else 0
            cumulative_return = (portfolio_value - self.initial_capital) / self.initial_capital
            
            # Benchmark
            current_spy_price = spy_data.loc[current_date, 'Close']
            benchmark_return = (current_spy_price - initial_spy_price) / initial_spy_price
            
            self.daily_snapshots.append(DailySnapshot(
                date=current_date,
                portfolio_value=portfolio_value,
                cash=self.cash,
                positions_value=portfolio_value - self.cash,
                num_positions=len(self.positions),
                daily_return=daily_return,
                cumulative_return=cumulative_return,
                benchmark_value=current_spy_price * (self.initial_capital / initial_spy_price),
                benchmark_return=benchmark_return,
                trades_today=trades_today
            ))
            
            prev_portfolio_value = portfolio_value
        
        # Close all remaining positions at end
        for ticker in list(self.positions.keys()):
            if ticker in historical_data:
                df = historical_data[ticker]
                final_price = df['Close'].iloc[-1]
                self._close_position(ticker, final_price, sim_end_date, "END_OF_SIMULATION")
        
        # Calculate final metrics
        return self._calculate_results(sim_start_date, sim_end_date, spy_data, initial_spy_price)
    
    def _calculate_results(
        self,
        start_date: datetime,
        end_date: datetime,
        spy_data: pd.DataFrame,
        initial_spy_price: float
    ) -> SimulationResult:
        """Calculate final simulation results and metrics."""
        
        final_value = self.cash
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # Annualized return
        days = (end_date - start_date).days
        years = days / 365.25
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Daily returns for Sharpe
        if self.daily_snapshots:
            daily_returns = pd.Series([s.daily_return for s in self.daily_snapshots])
            sharpe_ratio = (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0
            
            # Max drawdown
            equity = pd.Series([s.portfolio_value for s in self.daily_snapshots])
            rolling_max = equity.expanding().max()
            drawdowns = (equity - rolling_max) / rolling_max
            max_drawdown = drawdowns.min()
            
            # Equity curve
            equity_curve = pd.Series(
                [s.portfolio_value for s in self.daily_snapshots],
                index=[s.date for s in self.daily_snapshots]
            )
            benchmark_curve = pd.Series(
                [s.benchmark_value for s in self.daily_snapshots],
                index=[s.date for s in self.daily_snapshots]
            )
        else:
            sharpe_ratio = 0
            max_drawdown = 0
            equity_curve = pd.Series()
            benchmark_curve = pd.Series()
        
        # Trade statistics
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl < 0]
        
        win_rate = len(winning_trades) / len(self.trades) if self.trades else 0
        
        total_wins = sum(t.pnl for t in winning_trades)
        total_losses = abs(sum(t.pnl for t in losing_trades))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        avg_trade_pnl = sum(t.pnl for t in self.trades) / len(self.trades) if self.trades else 0
        avg_win = total_wins / len(winning_trades) if winning_trades else 0
        avg_loss = total_losses / len(losing_trades) if losing_trades else 0
        
        best_trade = max((t.pnl for t in self.trades), default=0)
        worst_trade = min((t.pnl for t in self.trades), default=0)
        
        # Benchmark comparison
        final_spy_price = spy_data['Close'].iloc[-1]
        benchmark_return = (final_spy_price - initial_spy_price) / initial_spy_price
        alpha = total_return - benchmark_return
        
        return SimulationResult(
            initial_capital=self.initial_capital,
            start_date=start_date,
            end_date=end_date,
            final_value=final_value,
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            benchmark_return=benchmark_return,
            alpha=alpha,
            total_trades=len(self.trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            avg_trade_pnl=avg_trade_pnl,
            avg_win=avg_win,
            avg_loss=avg_loss,
            best_trade=best_trade,
            worst_trade=worst_trade,
            trades=self.trades,
            daily_snapshots=self.daily_snapshots,
            equity_curve=equity_curve,
            benchmark_curve=benchmark_curve
        )
    
    def format_results(self, result: SimulationResult) -> str:
        """Format simulation results as a readable string."""
        # Determine if we beat the benchmark
        beat_benchmark = result.total_return > result.benchmark_return
        benchmark_status = "✅ BEAT" if beat_benchmark else "❌ MISSED"
        
        lines = [
            "=" * 70,
            "📊 TRADING SIMULATION RESULTS",
            "=" * 70,
            f"Period: {result.start_date.date()} to {result.end_date.date()}",
            f"Initial Capital: ${result.initial_capital:,.2f}",
            "",
            "-" * 70,
            "💰 PERFORMANCE",
            "-" * 70,
            f"Final Value:       ${result.final_value:,.2f}",
            f"Total Return:      {result.total_return:>10.2%}",
            f"Annualized Return: {result.annualized_return:>10.2%}",
            f"Sharpe Ratio:      {result.sharpe_ratio:>10.2f}",
            f"Max Drawdown:      {result.max_drawdown:>10.2%}",
            "",
            "-" * 70,
            f"📈 BENCHMARK COMPARISON (S&P 500) - {benchmark_status}",
            "-" * 70,
            f"S&P 500 Return:    {result.benchmark_return:>10.2%}",
            f"Strategy Return:   {result.total_return:>10.2%}",
            f"Alpha (Excess):    {result.alpha:>10.2%}",
            "",
            "-" * 70,
            "📋 TRADE STATISTICS",
            "-" * 70,
            f"Total Trades:      {result.total_trades:>10}",
            f"Winning Trades:    {result.winning_trades:>10}",
            f"Losing Trades:     {result.losing_trades:>10}",
            f"Win Rate:          {result.win_rate:>10.2%}",
            f"Profit Factor:     {result.profit_factor:>10.2f}",
            f"Avg Trade P&L:     ${result.avg_trade_pnl:>9.2f}",
            f"Avg Win:           ${result.avg_win:>9.2f}",
            f"Avg Loss:          ${result.avg_loss:>9.2f}",
            f"Best Trade:        ${result.best_trade:>9.2f}",
            f"Worst Trade:       ${result.worst_trade:>9.2f}",
            "=" * 70,
        ]
        
        return "\n".join(lines)
    
    def get_trade_log(self, result: SimulationResult, limit: int = 20) -> str:
        """Get a formatted log of trades."""
        lines = [
            "",
            "📜 TRADE LOG (Last {} trades)".format(min(limit, len(result.trades))),
            "-" * 90,
            f"{'Date':<12} {'Ticker':<8} {'Dir':<6} {'Entry':>10} {'Exit':>10} {'P&L':>12} {'Reason':<15}",
            "-" * 90,
        ]
        
        for trade in result.trades[-limit:]:
            entry_date = trade.entry_date.strftime('%Y-%m-%d') if trade.entry_date else 'N/A'
            pnl_color = '+' if trade.pnl >= 0 else ''
            lines.append(
                f"{entry_date:<12} {trade.ticker:<8} {trade.direction.value:<6} "
                f"${trade.entry_price:>9.2f} ${trade.exit_price:>9.2f} "
                f"{pnl_color}${trade.pnl:>10.2f} {trade.exit_reason:<15}"
            )
        
        lines.append("-" * 90)
        return "\n".join(lines)


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    
    print("Starting simulation test...")
    simulator = TradingSimulator(
        initial_capital=100000,
        max_positions=5,
        position_size_pct=0.20
    )
    
    # Use small set for testing
    test_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "AMD", "NFLX", "CRM"]
    
    def progress(day, total, info):
        print(f"Day {day}/{total}: {info['date'].date()} - Portfolio: ${info['portfolio_value']:,.2f}")
    
    result = simulator.run(
        tickers=test_tickers,
        period="2y",
        progress_callback=progress
    )
    
    print(simulator.format_results(result))
    print(simulator.get_trade_log(result))
