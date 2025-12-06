"""
Backtesting Module
Comprehensive backtesting framework with performance metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a single trade."""
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    position: int  # 1 for long, -1 for short
    pnl: float
    pnl_percent: float
    signal_strength: int


@dataclass
class BacktestResult:
    """Container for backtest results."""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_pnl: float
    avg_win: float
    avg_loss: float
    trades: List[Trade]
    equity_curve: pd.Series
    monthly_returns: pd.Series


class Backtester:
    """
    Backtesting engine for evaluating trading strategies.
    
    Features:
    - Walk-forward backtesting
    - Comprehensive metrics (Sharpe, Drawdown, Win Rate, etc.)
    - Trade-by-trade analysis
    - Equity curve generation
    """
    
    def __init__(
        self,
        initial_capital: float = 100000,
        position_size_pct: float = 1.0,
        commission: float = 0.001,  # 0.1% per trade
        slippage: float = 0.0005,   # 0.05% slippage
    ):
        """
        Initialize backtester.
        
        Args:
            initial_capital: Starting capital
            position_size_pct: Position size as fraction of capital (1.0 = 100%)
            commission: Commission per trade as fraction
            slippage: Slippage per trade as fraction
        """
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.commission = commission
        self.slippage = slippage
    
    def run(
        self,
        df: pd.DataFrame,
        signals: pd.Series,
        prices: Optional[pd.Series] = None
    ) -> BacktestResult:
        """
        Run backtest on given signals.
        
        Args:
            df: DataFrame with OHLCV data
            signals: Series with trading signals (-2, -1, 0, 1, 2)
            prices: Optional Series with execution prices (defaults to Close)
            
        Returns:
            BacktestResult with all metrics and trade history
        """
        if prices is None:
            prices = df['Close']
        
        # Align signals with prices
        signals = signals.reindex(prices.index).fillna(0)
        
        capital = self.initial_capital
        position = 0
        entry_price = 0
        entry_date = None
        signal_strength = 0
        
        trades: List[Trade] = []
        equity_curve = []
        dates = []
        
        for date, price in prices.items():
            signal = signals.get(date, 0)
            
            # Calculate current equity
            if position != 0:
                unrealized_pnl = position * (price - entry_price) * (capital / entry_price)
                current_equity = capital + unrealized_pnl
            else:
                current_equity = capital
            
            equity_curve.append(current_equity)
            dates.append(date)
            
            # Check for position changes
            if signal >= 1 and position <= 0:  # Buy signal
                # Close short if any
                if position < 0:
                    exit_price = price * (1 + self.slippage)  # Slippage for covering
                    pnl = position * (exit_price - entry_price) * (capital / entry_price)
                    pnl -= abs(pnl) * self.commission
                    capital += pnl
                    
                    trades.append(Trade(
                        entry_date=entry_date,
                        exit_date=date,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        position=-1,
                        pnl=pnl,
                        pnl_percent=pnl / capital * 100,
                        signal_strength=signal_strength
                    ))
                
                # Open long
                entry_price = price * (1 + self.slippage)
                entry_date = date
                position = 1
                signal_strength = int(signal)
                capital -= capital * self.commission * self.position_size_pct
                
            elif signal <= -1 and position >= 0:  # Sell signal
                # Close long if any
                if position > 0:
                    exit_price = price * (1 - self.slippage)
                    pnl = position * (exit_price - entry_price) * (capital / entry_price)
                    pnl -= abs(pnl) * self.commission
                    capital += pnl
                    
                    trades.append(Trade(
                        entry_date=entry_date,
                        exit_date=date,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        position=1,
                        pnl=pnl,
                        pnl_percent=pnl / capital * 100,
                        signal_strength=signal_strength
                    ))
                
                # Open short (only if strong sell)
                if signal <= -1:
                    entry_price = price * (1 - self.slippage)
                    entry_date = date
                    position = -1
                    signal_strength = int(signal)
                    capital -= capital * self.commission * self.position_size_pct
                else:
                    position = 0
            
            elif signal == 0 and position != 0:  # Flatten signal
                exit_price = price * (1 - self.slippage * position)
                pnl = position * (exit_price - entry_price) * (capital / entry_price)
                pnl -= abs(pnl) * self.commission
                capital += pnl
                
                trades.append(Trade(
                    entry_date=entry_date,
                    exit_date=date,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    position=position,
                    pnl=pnl,
                    pnl_percent=pnl / capital * 100,
                    signal_strength=signal_strength
                ))
                
                position = 0
        
        # Close any open position at end
        if position != 0:
            final_price = prices.iloc[-1]
            exit_price = final_price * (1 - self.slippage * position)
            pnl = position * (exit_price - entry_price) * (capital / entry_price)
            pnl -= abs(pnl) * self.commission
            capital += pnl
            
            trades.append(Trade(
                entry_date=entry_date,
                exit_date=prices.index[-1],
                entry_price=entry_price,
                exit_price=exit_price,
                position=position,
                pnl=pnl,
                pnl_percent=pnl / capital * 100,
                signal_strength=signal_strength
            ))
        
        # Create equity curve series
        equity_series = pd.Series(equity_curve, index=dates)
        
        # Calculate metrics
        return self._calculate_metrics(trades, equity_series)
    
    def _calculate_metrics(
        self,
        trades: List[Trade],
        equity_curve: pd.Series
    ) -> BacktestResult:
        """Calculate performance metrics from trade history."""
        
        # Total return
        total_return = (equity_curve.iloc[-1] - self.initial_capital) / self.initial_capital
        
        # Annualized return
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        years = days / 365.25
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Daily returns for Sharpe
        daily_returns = equity_curve.pct_change().dropna()
        
        # Sharpe Ratio (assuming 0% risk-free rate)
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            sharpe_ratio = (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252))
        else:
            sharpe_ratio = 0
        
        # Maximum Drawdown
        rolling_max = equity_curve.expanding().max()
        drawdowns = (equity_curve - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        # Trade statistics
        if len(trades) > 0:
            winning_trades = [t for t in trades if t.pnl > 0]
            losing_trades = [t for t in trades if t.pnl < 0]
            
            win_rate = len(winning_trades) / len(trades)
            
            total_wins = sum(t.pnl for t in winning_trades)
            total_losses = abs(sum(t.pnl for t in losing_trades))
            
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
            avg_trade_pnl = sum(t.pnl for t in trades) / len(trades)
            avg_win = total_wins / len(winning_trades) if winning_trades else 0
            avg_loss = total_losses / len(losing_trades) if losing_trades else 0
        else:
            win_rate = 0
            profit_factor = 0
            avg_trade_pnl = 0
            avg_win = 0
            avg_loss = 0
        
        # Monthly returns
        monthly_returns = equity_curve.resample('M').last().pct_change().dropna()
        
        return BacktestResult(
            total_return=total_return,
            annualized_return=annualized_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(trades),
            avg_trade_pnl=avg_trade_pnl,
            avg_win=avg_win,
            avg_loss=avg_loss,
            trades=trades,
            equity_curve=equity_curve,
            monthly_returns=monthly_returns
        )
    
    def generate_signals_from_predictions(
        self,
        df: pd.DataFrame,
        predictor,
        indicators
    ) -> pd.Series:
        """
        Generate trading signals by running predictor on historical data.
        Uses walk-forward approach to avoid look-ahead bias.
        
        Args:
            df: DataFrame with OHLCV data
            predictor: MLPredictor instance
            indicators: TechnicalIndicators class
            
        Returns:
            Series with signals for each date
        """
        signals = pd.Series(index=df.index, dtype=int)
        signals[:] = 0
        
        # Minimum training window
        min_window = 252  # 1 year
        
        for i in range(min_window, len(df)):
            # Use data up to point i for training/prediction
            train_df = df.iloc[:i]
            
            # Calculate indicators
            ti = indicators(train_df)
            df_with_indicators = ti.get_dataframe()
            
            # Retrain model periodically (every 20 days)
            if i == min_window or i % 20 == 0:
                predictor.train(df_with_indicators, n_splits=3)
            
            # Make prediction
            result = predictor.predict(df_with_indicators)
            
            if result:
                signals.iloc[i] = result.prediction.value
        
        return signals
    
    def format_results(self, result: BacktestResult) -> str:
        """Format backtest results as readable string."""
        lines = [
            "=" * 50,
            "BACKTEST RESULTS",
            "=" * 50,
            f"Total Return:       {result.total_return:>10.2%}",
            f"Annualized Return:  {result.annualized_return:>10.2%}",
            f"Sharpe Ratio:       {result.sharpe_ratio:>10.2f}",
            f"Max Drawdown:       {result.max_drawdown:>10.2%}",
            "-" * 50,
            f"Total Trades:       {result.total_trades:>10}",
            f"Win Rate:           {result.win_rate:>10.2%}",
            f"Profit Factor:      {result.profit_factor:>10.2f}",
            f"Avg Trade P&L:      ${result.avg_trade_pnl:>9.2f}",
            f"Avg Win:            ${result.avg_win:>9.2f}",
            f"Avg Loss:           ${result.avg_loss:>9.2f}",
            "=" * 50,
        ]
        return "\n".join(lines)


if __name__ == "__main__":
    # Quick test with synthetic signals
    import yfinance as yf
    
    print("Fetching AAPL data...")
    stock = yf.Ticker("AAPL")
    df = stock.history(period="2y")
    
    # Create simple momentum signals for testing
    signals = pd.Series(index=df.index, dtype=int)
    returns = df['Close'].pct_change(5)
    signals[returns > 0.02] = 1
    signals[returns > 0.05] = 2
    signals[returns < -0.02] = -1
    signals[returns < -0.05] = -2
    signals = signals.fillna(0)
    
    print("Running backtest...")
    backtester = Backtester(initial_capital=100000)
    result = backtester.run(df, signals)
    
    print(backtester.format_results(result))
