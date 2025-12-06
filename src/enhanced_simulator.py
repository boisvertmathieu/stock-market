"""
Enhanced Trading Simulator
Uses the improved predictor with optimized strategy parameters.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging

from .data_fetcher import DataFetcher, TOP_100_TICKERS
from .indicators import TechnicalIndicators
from .enhanced_predictor import EnhancedMLPredictor

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
    trailing_stop: float = 0.0  # For trailing stop loss


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
    initial_capital: float
    start_date: datetime
    end_date: datetime
    final_value: float
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    benchmark_return: float
    alpha: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_trade_pnl: float
    avg_win: float
    avg_loss: float
    best_trade: float
    worst_trade: float
    trades: List[Trade] = field(default_factory=list)
    daily_snapshots: List[DailySnapshot] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    benchmark_curve: pd.Series = field(default_factory=pd.Series)


class EnhancedSimulator:
    """
    Enhanced trading simulator with improved strategy:
    - Dynamic signal thresholds based on market regime
    - Trailing stop losses
    - Position sizing based on conviction
    - Trend-following bias
    """
    
    def __init__(
        self,
        initial_capital: float = 100000,
        max_positions: int = 10,
        base_position_size: float = 0.10,
        commission: float = 0.001,
        slippage: float = 0.0005,
        signal_threshold: float = 0.3,
        use_trailing_stop: bool = True,
        silent: bool = True,
    ):
        self.initial_capital = initial_capital
        self.max_positions = max_positions
        self.base_position_size = base_position_size
        self.commission = commission
        self.slippage = slippage
        self.signal_threshold = signal_threshold
        self.use_trailing_stop = use_trailing_stop
        self.silent = silent
        
        if silent:
            for logger_name in ['src.predictor', 'src.enhanced_predictor', 
                               'src.indicators', 'src.data_fetcher', 'optuna']:
                logging.getLogger(logger_name).setLevel(logging.CRITICAL)
        
        self.fetcher = DataFetcher()
        self.predictors: Dict[str, EnhancedMLPredictor] = {}
        
        # State
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.daily_snapshots: List[DailySnapshot] = []
        self.market_trend = 0.0  # Bull/Bear indicator
        
    def _get_predictor(self, ticker: str) -> EnhancedMLPredictor:
        if ticker not in self.predictors:
            self.predictors[ticker] = EnhancedMLPredictor(
                optimize_hyperparams=False,  # Speed up for simulation
                use_shap=False
            )
        return self.predictors[ticker]
    
    def _get_fundamental_data(self, ticker: str) -> Optional[Dict]:
        """Fetch fundamental data from yfinance."""
        try:
            import yfinance as yf
            stock = yf.Ticker(ticker)
            info = stock.info
            return {
                'pe_ratio': info.get('forwardPE') or info.get('trailingPE'),
                'peg_ratio': info.get('pegRatio'),
                'profit_margin': info.get('profitMargins'),
                'revenue_growth': info.get('revenueGrowth'),
                'market_cap': info.get('marketCap'),
                'sector': info.get('sector'),
            }
        except:
            return None
    
    def _analyze_ticker(
        self,
        ticker: str,
        historical_data: Dict[str, pd.DataFrame],
        current_date: datetime,
        fundamental_data: Optional[Dict] = None
    ) -> Optional[Dict[str, Any]]:
        """Analyze ticker with enhanced features."""
        if ticker not in historical_data:
            return None
        
        df = historical_data[ticker]
        available_data = df[df.index <= current_date]
        
        if len(available_data) < 100:
            return None
        
        analysis_data = available_data.tail(300)
        
        try:
            ti = TechnicalIndicators(analysis_data)
            analysis = ti.get_comprehensive_analysis()
            
            current_price = analysis_data['Close'].iloc[-1]
            atr = analysis['raw_indicators'].get('ATR', current_price * 0.02)
            
            # Technical score
            tech_score = analysis['overall']['signal_value']
            
            # ML prediction
            predictor = self._get_predictor(ticker)
            ml_score = 0
            ml_pred_name = 'N/A'
            ml_confidence = 0.5
            
            if predictor.is_trained:
                try:
                    df_ind = ti.get_dataframe()
                    result = predictor.predict(df_ind, fundamental_data)
                    if result:
                        ml_score = result.prediction.value
                        ml_pred_name = result.prediction.name
                        ml_confidence = result.probability
                except:
                    pass
            
            # Adaptive scoring based on market regime
            if self.market_trend > 0.5:
                # Bull market: favor technical + momentum
                combined_score = 0.6 * tech_score + 0.4 * ml_score
            elif self.market_trend < -0.5:
                # Bear market: favor ML + caution
                combined_score = 0.4 * tech_score + 0.6 * ml_score
            else:
                # Neutral: equal weight
                combined_score = 0.5 * tech_score + 0.5 * ml_score
            
            # Boost score for high conviction ML predictions
            if ml_confidence > 0.7:
                combined_score *= 1.2
            
            return {
                'ticker': ticker,
                'date': current_date,
                'price': current_price,
                'technical_signal': analysis['overall']['signal'],
                'technical_score': tech_score,
                'ml_prediction': ml_pred_name,
                'ml_score': ml_score,
                'ml_confidence': ml_confidence,
                'combined_score': combined_score,
                'atr': atr,
                'volume_ratio': analysis['raw_indicators'].get('Volume_Ratio', 1.0),
            }
        except:
            return None
    
    def _calculate_position_size(self, signal_score: float, atr: float, price: float) -> float:
        """Dynamic position sizing based on conviction and volatility."""
        # Base size adjusted by signal strength
        conviction_mult = min(1.5, max(0.5, abs(signal_score) / 1.5))
        
        # Reduce size for volatile stocks
        volatility_adj = min(1.0, 0.02 / (atr / price + 0.001))
        
        return self.base_position_size * conviction_mult * volatility_adj
    
    def _execute_trade(
        self,
        ticker: str,
        direction: TradeDirection,
        price: float,
        date: datetime,
        atr: float,
        signal_score: float
    ) -> bool:
        if len(self.positions) >= self.max_positions:
            return False
        if ticker in self.positions:
            return False
        
        position_pct = self._calculate_position_size(signal_score, atr, price)
        position_value = self.cash * position_pct
        if position_value <= 0:
            return False
        
        entry_price = price * (1 + self.slippage if direction == TradeDirection.LONG else 1 - self.slippage)
        
        # Dynamic stop based on ATR and market trend
        # More aggressive in bull market (wider stops)
        if self.market_trend > 0.5:
            atr_mult = 2.5  # Bull: give more room
            tp_mult = 4.0   # Bull: higher targets
        elif self.market_trend < -0.5:
            atr_mult = 1.5  # Bear: tighter stops
            tp_mult = 2.0   # Bear: take profits quicker
        else:
            atr_mult = 2.0
            tp_mult = 3.0
        
        if direction == TradeDirection.LONG:
            stop_loss = entry_price - atr_mult * atr
            take_profit = entry_price + tp_mult * atr
        else:
            stop_loss = entry_price + atr_mult * atr
            take_profit = entry_price - atr_mult * 2 * atr
        
        shares = int(position_value / entry_price)
        if shares <= 0:
            return False
        
        cost = shares * entry_price * (1 + self.commission)
        if cost > self.cash:
            shares = int((self.cash * 0.95 / (1 + self.commission)) / entry_price)
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
            signal_score=signal_score,
            trailing_stop=stop_loss
        )
        
        return True
    
    def _update_trailing_stops(self, ticker: str, current_price: float):
        """Update trailing stop for position."""
        if not self.use_trailing_stop or ticker not in self.positions:
            return
        
        pos = self.positions[ticker]
        if pos.direction == TradeDirection.LONG:
            # Move stop up if price moves favorably
            # Wide trailing stop (10%) to maximize trend capture
            new_stop = current_price * 0.90
            if new_stop > pos.trailing_stop:
                pos.trailing_stop = new_stop
    
    def _close_position(self, ticker: str, price: float, date: datetime, reason: str) -> Optional[Trade]:
        if ticker not in self.positions:
            return None
        
        pos = self.positions[ticker]
        exit_price = price * (1 - self.slippage if pos.direction == TradeDirection.LONG else 1 + self.slippage)
        
        if pos.direction == TradeDirection.LONG:
            pnl = (exit_price - pos.entry_price) * pos.shares
        else:
            pnl = (pos.entry_price - exit_price) * pos.shares
        
        pnl -= exit_price * pos.shares * self.commission
        pnl_percent = pnl / (pos.entry_price * pos.shares) * 100
        
        self.cash += exit_price * pos.shares * (1 - self.commission)
        
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
    
    def _check_exits(self, ticker: str, current_price: float, date: datetime) -> bool:
        if ticker not in self.positions:
            return False
        
        pos = self.positions[ticker]
        
        # Update trailing stop
        self._update_trailing_stops(ticker, current_price)
        
        if pos.direction == TradeDirection.LONG:
            # Check trailing stop
            if self.use_trailing_stop and current_price <= pos.trailing_stop:
                self._close_position(ticker, current_price, date, "TRAILING_STOP")
                return True
            if current_price <= pos.stop_loss:
                self._close_position(ticker, current_price, date, "STOP_LOSS")
                return True
            if current_price >= pos.take_profit:
                self._close_position(ticker, current_price, date, "TAKE_PROFIT")
                return True
        
        return False
    
    def _update_market_trend(self, spy_data: pd.DataFrame, current_date: datetime):
        """Calculate market trend from SPY for regime detection."""
        available = spy_data[spy_data.index <= current_date]
        if len(available) < 50:
            self.market_trend = 0
            return
        
        # 50-day return
        ret_50 = (available['Close'].iloc[-1] / available['Close'].iloc[-50] - 1)
        # 20-day return
        ret_20 = (available['Close'].iloc[-1] / available['Close'].iloc[-20] - 1)
        
        # MA trend
        sma_20 = available['Close'].rolling(20).mean().iloc[-1]
        sma_50 = available['Close'].rolling(50).mean().iloc[-1]
        
        # Composite trend score
        self.market_trend = 0.3 * np.sign(ret_50) + 0.3 * np.sign(ret_20) + 0.4 * np.sign(sma_20 - sma_50)
    
    def _calculate_portfolio_value(self, historical_data: Dict[str, pd.DataFrame], date: datetime) -> float:
        positions_value = 0.0
        for ticker, pos in self.positions.items():
            if ticker in historical_data:
                df = historical_data[ticker]
                available = df[df.index <= date]
                if len(available) > 0:
                    current_price = available['Close'].iloc[-1]
                    positions_value += current_price * pos.shares
        return self.cash + positions_value
    
    def run(
        self,
        tickers: Optional[List[str]] = None,
        period: str = "2y",
        progress_callback = None
    ) -> SimulationResult:
        """Run the enhanced simulation."""
        if tickers is None:
            tickers = TOP_100_TICKERS[:30]
        
        # Reset state
        self.cash = self.initial_capital
        self.positions = {}
        self.trades = []
        self.daily_snapshots = []
        self.predictors = {}
        
        # Fetch data
        historical_data = {}
        fundamental_data = {}
        
        for ticker in tickers + ['SPY']:
            df = self.fetcher.fetch_ticker_data(ticker, period=period)
            if df is not None and len(df) > 100:
                historical_data[ticker] = df
                # Get fundamentals once
                if ticker != 'SPY':
                    fundamental_data[ticker] = self._get_fundamental_data(ticker)
        
        if 'SPY' not in historical_data:
            raise ValueError("Could not fetch SPY data")
        
        spy_data = historical_data['SPY']
        all_dates = spy_data.index.tolist()
        
        # Simulation period
        sim_days = min(500, len(all_dates) - 100)
        trading_dates = all_dates[-sim_days:]
        
        sim_start = trading_dates[0]
        sim_end = trading_dates[-1]
        initial_spy = spy_data.loc[sim_start, 'Close']
        
        # Pre-train predictors
        warmup_date = trading_dates[0]
        for ticker in tickers:
            if ticker in historical_data:
                df = historical_data[ticker]
                warmup = df[df.index < warmup_date]
                if len(warmup) >= 100:
                    try:
                        ti = TechnicalIndicators(warmup.tail(300))
                        df_ind = ti.get_dataframe()
                        predictor = self._get_predictor(ticker)
                        predictor.train(df_ind, fundamental_data.get(ticker), n_optuna_trials=10)
                    except:
                        pass
        
        # Run simulation
        prev_value = self.initial_capital
        
        for day_idx, current_date in enumerate(trading_dates):
            trades_today = 0
            
            if progress_callback and day_idx % 10 == 0:
                progress_callback(day_idx, len(trading_dates), {
                    'date': current_date,
                    'portfolio_value': prev_value,
                    'positions': len(self.positions),
                    'trades': len(self.trades)
                })
            
            # Update market regime
            self._update_market_trend(spy_data, current_date)
            
            # Check exits
            for ticker in list(self.positions.keys()):
                if ticker in historical_data:
                    df = historical_data[ticker]
                    avail = df[df.index <= current_date]
                    if len(avail) > 0:
                        if self._check_exits(ticker, avail['Close'].iloc[-1], current_date):
                            trades_today += 1
            
            # Analyze tickers
            analyses = []
            for ticker in tickers:
                if ticker in historical_data:
                    analysis = self._analyze_ticker(ticker, historical_data, current_date, 
                                                   fundamental_data.get(ticker))
                    if analysis:
                        analyses.append(analysis)
            
            # Filter by signal threshold
            buy_signals = [a for a in analyses if a['combined_score'] >= self.signal_threshold]
            buy_signals.sort(key=lambda x: -x['combined_score'])
            
            # Signal reversal exits
            for ticker in list(self.positions.keys()):
                pos = self.positions[ticker]
                ticker_analysis = next((a for a in analyses if a['ticker'] == ticker), None)
                if ticker_analysis and pos.direction == TradeDirection.LONG:
                    if ticker_analysis['combined_score'] <= -0.3:
                        if ticker in historical_data:
                            df = historical_data[ticker]
                            avail = df[df.index <= current_date]
                            if len(avail) > 0:
                                self._close_position(ticker, avail['Close'].iloc[-1], current_date, "SIGNAL_REVERSE")
                                trades_today += 1
            
            # Open new positions
            for candidate in buy_signals[:5]:
                if len(self.positions) < self.max_positions:
                    if candidate['ticker'] not in self.positions:
                        if self._execute_trade(
                            candidate['ticker'],
                            TradeDirection.LONG,
                            candidate['price'],
                            current_date,
                            candidate['atr'],
                            candidate['combined_score']
                        ):
                            trades_today += 1
            
            # Snapshot
            portfolio_value = self._calculate_portfolio_value(historical_data, current_date)
            daily_return = (portfolio_value - prev_value) / prev_value if prev_value > 0 else 0
            cum_return = (portfolio_value - self.initial_capital) / self.initial_capital
            
            spy_price = spy_data.loc[current_date, 'Close']
            bench_return = (spy_price - initial_spy) / initial_spy
            
            self.daily_snapshots.append(DailySnapshot(
                date=current_date,
                portfolio_value=portfolio_value,
                cash=self.cash,
                positions_value=portfolio_value - self.cash,
                num_positions=len(self.positions),
                daily_return=daily_return,
                cumulative_return=cum_return,
                benchmark_value=spy_price * (self.initial_capital / initial_spy),
                benchmark_return=bench_return,
                trades_today=trades_today
            ))
            
            prev_value = portfolio_value
        
        # Close remaining
        for ticker in list(self.positions.keys()):
            if ticker in historical_data:
                self._close_position(ticker, historical_data[ticker]['Close'].iloc[-1], sim_end, "END")
        
        return self._calculate_results(sim_start, sim_end, spy_data, initial_spy)
    
    def _calculate_results(self, start, end, spy_data, initial_spy) -> SimulationResult:
        final_value = self.cash
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        days = (end - start).days
        years = days / 365.25
        ann_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        if self.daily_snapshots:
            daily_rets = pd.Series([s.daily_return for s in self.daily_snapshots])
            sharpe = (daily_rets.mean() * 252) / (daily_rets.std() * np.sqrt(252)) if daily_rets.std() > 0 else 0
            
            equity = pd.Series([s.portfolio_value for s in self.daily_snapshots])
            max_dd = ((equity - equity.expanding().max()) / equity.expanding().max()).min()
        else:
            sharpe = 0
            max_dd = 0
        
        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl < 0]
        
        win_rate = len(wins) / len(self.trades) if self.trades else 0
        total_wins = sum(t.pnl for t in wins)
        total_losses = abs(sum(t.pnl for t in losses))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        bench_return = (spy_data['Close'].iloc[-1] - initial_spy) / initial_spy
        
        return SimulationResult(
            initial_capital=self.initial_capital,
            start_date=start,
            end_date=end,
            final_value=final_value,
            total_return=total_return,
            annualized_return=ann_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            profit_factor=profit_factor,
            benchmark_return=bench_return,
            alpha=total_return - bench_return,
            total_trades=len(self.trades),
            winning_trades=len(wins),
            losing_trades=len(losses),
            avg_trade_pnl=sum(t.pnl for t in self.trades) / len(self.trades) if self.trades else 0,
            avg_win=total_wins / len(wins) if wins else 0,
            avg_loss=total_losses / len(losses) if losses else 0,
            best_trade=max((t.pnl for t in self.trades), default=0),
            worst_trade=min((t.pnl for t in self.trades), default=0),
            trades=self.trades,
            daily_snapshots=self.daily_snapshots,
        )
    
    def format_results(self, result: SimulationResult) -> str:
        beat = "✅ BEAT" if result.total_return > result.benchmark_return else "❌ MISSED"
        
        return f"""
{'='*70}
📊 ENHANCED TRADING SIMULATION RESULTS
{'='*70}
Period: {result.start_date.date()} to {result.end_date.date()}
Initial Capital: ${result.initial_capital:,.2f}

{'-'*70}
💰 PERFORMANCE
{'-'*70}
Final Value:       ${result.final_value:,.2f}
Total Return:      {result.total_return:>10.2%}
Annualized Return: {result.annualized_return:>10.2%}
Sharpe Ratio:      {result.sharpe_ratio:>10.2f}
Max Drawdown:      {result.max_drawdown:>10.2%}

{'-'*70}
📈 BENCHMARK COMPARISON (S&P 500) - {beat}
{'-'*70}
S&P 500 Return:    {result.benchmark_return:>10.2%}
Strategy Return:   {result.total_return:>10.2%}
Alpha (Excess):    {result.alpha:>10.2%}

{'-'*70}
📋 TRADE STATISTICS
{'-'*70}
Total Trades:      {result.total_trades:>10}
Winning Trades:    {result.winning_trades:>10}
Losing Trades:     {result.losing_trades:>10}
Win Rate:          {result.win_rate:>10.2%}
Profit Factor:     {result.profit_factor:>10.2f}
Avg Trade P&L:     ${result.avg_trade_pnl:>9.2f}
Best Trade:        ${result.best_trade:>9.2f}
Worst Trade:       ${result.worst_trade:>9.2f}
{'='*70}
"""
    
    def get_trade_log(self, result: SimulationResult, limit: int = 20) -> str:
        """Get formatted trade log."""
        lines = [
            "",
            f"📜 TRADE LOG (Last {min(limit, len(result.trades))} trades)",
            "-" * 90,
            f"{'Date':<12} {'Ticker':<8} {'Dir':<6} {'Entry':>10} {'Exit':>10} {'P&L':>12} {'Reason':<15}",
            "-" * 90,
        ]
        
        for trade in result.trades[-limit:]:
            entry_date = trade.entry_date.strftime('%Y-%m-%d') if trade.entry_date else 'N/A'
            pnl_prefix = '+' if trade.pnl >= 0 else ''
            lines.append(
                f"{entry_date:<12} {trade.ticker:<8} {trade.direction.value:<6} "
                f"${trade.entry_price:>9.2f} ${trade.exit_price:>9.2f} "
                f"{pnl_prefix}${trade.pnl:>10.2f} {trade.exit_reason:<15}"
            )
        
        lines.append("-" * 90)
        return "\n".join(lines)
