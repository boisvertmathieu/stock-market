"""
Live Trading Simulation
Persistent trading simulation that runs on schedule (every 2h).
Tracks portfolio state, analyzes market, and recommends trades.
"""

import os
import gc
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging

from .data_fetcher import DataFetcher, TOP_100_TICKERS
from .indicators import TechnicalIndicators
from .finnhub_client import FinnHubClient, get_finnhub_client

logger = logging.getLogger(__name__)

# Default state file location
STATE_FILE = os.getenv("LONGRUN_STATE_FILE", "./data/longrun_state.json")


class TradeAction(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Position:
    """Active position in portfolio."""
    ticker: str
    shares: int
    entry_price: float
    entry_date: str
    stop_loss: float
    take_profit: float
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0


@dataclass
class Trade:
    """Completed trade."""
    ticker: str
    action: str
    shares: int
    price: float
    timestamp: str
    pnl: float = 0.0
    reason: str = ""


@dataclass
class TradeRecommendation:
    """Recommended trade action."""
    ticker: str
    action: TradeAction
    confidence: float  # 0-1
    current_price: float
    target_shares: int
    reason: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


@dataclass
class PortfolioState:
    """Complete portfolio state for persistence."""
    initialized_at: str
    initial_capital: float
    current_cash: float
    positions: List[Dict] = field(default_factory=list)
    trade_history: List[Dict] = field(default_factory=list)
    last_run: str = ""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    
    @property
    def positions_value(self) -> float:
        return sum(p.get('shares', 0) * p.get('current_price', p.get('entry_price', 0)) 
                   for p in self.positions)
    
    @property
    def total_value(self) -> float:
        return self.current_cash + self.positions_value
    
    @property
    def total_return(self) -> float:
        return (self.total_value - self.initial_capital) / self.initial_capital
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PortfolioState':
        return cls(**data)
    
    def to_dict(self) -> Dict:
        return asdict(self)


class LiveTrader:
    """
    Live trading simulation engine.
    Runs on schedule, persists state, and recommends trades.
    """
    
    def __init__(
        self,
        state_file: str = STATE_FILE,
        tickers: Optional[List[str]] = None,
        max_positions: int = 10,
        position_size_pct: float = 0.10,
        stop_loss_pct: float = 0.08,
        take_profit_pct: float = 0.15,
    ):
        self.state_file = Path(state_file)
        # Reduced from 30 to 15 for memory optimization on Raspberry Pi
        self.tickers = tickers or TOP_100_TICKERS[:15]
        self.max_positions = max_positions
        self.position_size_pct = position_size_pct
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        self.fetcher = DataFetcher()
        self.finnhub = get_finnhub_client()
        self.state: Optional[PortfolioState] = None
        
    def initialize(self, initial_capital: float) -> PortfolioState:
        """Initialize a new trading simulation."""
        self.state = PortfolioState(
            initialized_at=datetime.now().isoformat(),
            initial_capital=initial_capital,
            current_cash=initial_capital,
            positions=[],
            trade_history=[],
            last_run=datetime.now().isoformat(),
        )
        self._save_state()
        logger.info(f"Initialized new simulation with ${initial_capital:,.2f}")
        return self.state
    
    def load_state(self) -> Optional[PortfolioState]:
        """Load existing state from file."""
        if not self.state_file.exists():
            return None
        
        try:
            with open(self.state_file, 'r') as f:
                data = json.load(f)
            self.state = PortfolioState.from_dict(data)
            logger.info(f"Loaded state: ${self.state.total_value:,.2f} total value")
            return self.state
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return None
    
    def _save_state(self):
        """Save state to file."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(self.state.to_dict(), f, indent=2)
        logger.debug(f"Saved state to {self.state_file}")
    
    def _get_market_data(self, ticker: str) -> Optional[Dict]:
        """Get combined market data from yFinance + FinnHub."""
        # Technical data from yFinance
        df = self.fetcher.fetch_ticker_data(ticker, period="60d")
        if df is None or len(df) < 20:
            return None
        
        try:
            ti = TechnicalIndicators(df)
            analysis = ti.get_comprehensive_analysis()
            current_price = df['Close'].iloc[-1]
            atr = analysis['raw_indicators'].get('ATR', current_price * 0.02)
            
            # FinnHub data
            fh_data = self.finnhub._cache.get(ticker)
            
            return {
                'ticker': ticker,
                'current_price': current_price,
                'technical_signal': analysis['overall']['signal_value'],
                'technical_name': analysis['overall']['signal'],
                'atr': atr,
                'volume_ratio': analysis['raw_indicators'].get('Volume_Ratio', 1.0),
                'rsi': analysis['raw_indicators'].get('RSI', 50),
                'finnhub': {
                    'analyst_score': fh_data.analyst_score if fh_data else 0,
                    'value_score': fh_data.value_score if fh_data else 0,
                    'pe_ratio': fh_data.pe_ratio if fh_data else None,
                    'change_percent': fh_data.change_percent if fh_data else 0,
                    'sentiment_score': fh_data.sentiment_score if fh_data else 0,
                } if fh_data else None
            }
        except Exception as e:
            logger.warning(f"Failed to analyze {ticker}: {e}")
            return None
    
    def _calculate_signal_score(self, data: Dict) -> float:
        """Calculate combined signal score including sentiment."""
        # Base technical score (45%)
        tech_score = data['technical_signal'] * 0.45
        
        # FinnHub data (35% fundamental/analyst + 20% sentiment)
        fh = data.get('finnhub')
        if fh:
            analyst = fh.get('analyst_score', 0) * 0.20
            value = fh.get('value_score', 0) * 0.10
            momentum = 0.05 if fh.get('change_percent', 0) > 0 else -0.03
            
            # Sentiment score (20% weight)
            sentiment = fh.get('sentiment_score', 0) * 0.20
            
            return tech_score + analyst + value + momentum + sentiment
        
        return tech_score
    
    def _update_positions(self, market_data: Dict[str, Dict]):
        """Update current prices and P&L for positions."""
        for pos in self.state.positions:
            ticker = pos['ticker']
            if ticker in market_data:
                current = market_data[ticker]['current_price']
                pos['current_price'] = current
                pos['unrealized_pnl'] = (current - pos['entry_price']) * pos['shares']
                pos['unrealized_pnl_pct'] = (current / pos['entry_price'] - 1) * 100
    
    def _check_exits(self, market_data: Dict[str, Dict]) -> List[TradeRecommendation]:
        """Check if any positions should be closed."""
        recommendations = []
        
        for pos in self.state.positions:
            ticker = pos['ticker']
            if ticker not in market_data:
                continue
            
            current = market_data[ticker]['current_price']
            entry = pos['entry_price']
            
            # Stop loss
            if current <= pos['stop_loss']:
                recommendations.append(TradeRecommendation(
                    ticker=ticker,
                    action=TradeAction.SELL,
                    confidence=1.0,
                    current_price=current,
                    target_shares=pos['shares'],
                    reason=f"STOP LOSS hit (entry: ${entry:.2f}, stop: ${pos['stop_loss']:.2f})"
                ))
            # Take profit
            elif current >= pos['take_profit']:
                recommendations.append(TradeRecommendation(
                    ticker=ticker,
                    action=TradeAction.SELL,
                    confidence=1.0,
                    current_price=current,
                    target_shares=pos['shares'],
                    reason=f"TAKE PROFIT hit (entry: ${entry:.2f}, target: ${pos['take_profit']:.2f})"
                ))
            # Signal reversal
            elif market_data[ticker]['technical_signal'] < -0.5:
                recommendations.append(TradeRecommendation(
                    ticker=ticker,
                    action=TradeAction.SELL,
                    confidence=0.8,
                    current_price=current,
                    target_shares=pos['shares'],
                    reason="Signal reversal (bearish)"
                ))
        
        return recommendations
    
    def _find_entries(self, market_data: Dict[str, Dict]) -> List[TradeRecommendation]:
        """Find new entry opportunities."""
        recommendations = []
        current_tickers = {p['ticker'] for p in self.state.positions}
        available_slots = self.max_positions - len(current_tickers)
        
        if available_slots <= 0:
            return []
        
        # Score all tickers
        scored = []
        for ticker, data in market_data.items():
            if ticker in current_tickers:
                continue
            
            score = self._calculate_signal_score(data)
            if score > 0.5:  # Bullish threshold
                scored.append((ticker, score, data))
        
        # Sort by score and take top opportunities
        scored.sort(key=lambda x: -x[1])
        
        for ticker, score, data in scored[:available_slots]:
            price = data['current_price']
            atr = data['atr']
            
            # Calculate position size
            position_value = self.state.current_cash * self.position_size_pct
            shares = int(position_value / price)
            
            if shares <= 0:
                continue
            
            # Calculate stops
            stop_loss = price - 2 * atr
            take_profit = price + 3 * atr
            
            reason_parts = [f"Score: {score:.2f}"]
            if data.get('finnhub'):
                fh = data['finnhub']
                if fh.get('analyst_score', 0) > 0.3:
                    reason_parts.append(f"Strong analyst ({fh['analyst_score']:.2f})")
                if fh.get('pe_ratio'):
                    reason_parts.append(f"P/E: {fh['pe_ratio']:.1f}")
            
            recommendations.append(TradeRecommendation(
                ticker=ticker,
                action=TradeAction.BUY,
                confidence=min(1.0, score),
                current_price=price,
                target_shares=shares,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=" | ".join(reason_parts)
            ))
        
        return recommendations
    
    def run_cycle(self, execute: bool = False) -> Dict[str, Any]:
        """
        Run a single trading cycle.
        
        Args:
            execute: If True, actually execute the trades. If False, just recommend.
            
        Returns:
            Dict with recommendations and portfolio status
        """
        try:
            if not self.state:
                self.load_state()
            
            if not self.state:
                raise ValueError("No state found. Initialize first with --init")
            
            # Process tickers in batches to reduce memory usage (Raspberry Pi optimization)
            batch_size = 5
            market_data = {}
            
            for batch_start in range(0, len(self.tickers), batch_size):
                batch = self.tickers[batch_start:batch_start + batch_size]
                
                # Fetch FinnHub data for this batch only
                self.finnhub.fetch_all(batch)
                
                # Get market data for batch
                for ticker in batch:
                    data = self._get_market_data(ticker)
                    if data:
                        market_data[ticker] = data
                
                # Force garbage collection between batches to free memory
                gc.collect()
            
            # Update positions with current prices
            self._update_positions(market_data)
            
            # Get recommendations
            sell_recs = self._check_exits(market_data)
            buy_recs = self._find_entries(market_data)
            
            # Execute if requested
            executed_trades = []
            if execute:
                # Execute sells first
                for rec in sell_recs:
                    trade = self._execute_sell(rec)
                    if trade:
                        executed_trades.append(trade)
                
                # Then buys
                for rec in buy_recs:
                    trade = self._execute_buy(rec)
                    if trade:
                        executed_trades.append(trade)
                
                self.state.last_run = datetime.now().isoformat()
                self._save_state()
            
            return {
                'timestamp': datetime.now().isoformat(),
                'portfolio': {
                    'total_value': self.state.total_value,
                    'cash': self.state.current_cash,
                    'positions_value': self.state.positions_value,
                    'total_return': self.state.total_return,
                    'positions_count': len(self.state.positions),
                },
                'recommendations': {
                    'sell': [asdict(r) for r in sell_recs],
                    'buy': [asdict(r) for r in buy_recs],
                },
                'executed_trades': executed_trades,
                'positions': self.state.positions,
            }
        except Exception as e:
            logger.error(f"Trading cycle failed: {e}")
            # Return error state instead of crashing silently
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'portfolio': {
                    'total_value': self.state.total_value if self.state else 0,
                    'cash': self.state.current_cash if self.state else 0,
                    'positions_value': self.state.positions_value if self.state else 0,
                    'total_return': self.state.total_return if self.state else 0,
                    'positions_count': len(self.state.positions) if self.state else 0,
                },
                'recommendations': {'sell': [], 'buy': []},
                'executed_trades': [],
                'positions': self.state.positions if self.state else [],
            }
    
    def _execute_buy(self, rec: TradeRecommendation) -> Optional[Dict]:
        """Execute a buy trade."""
        cost = rec.current_price * rec.target_shares * 1.001  # 0.1% commission
        
        if cost > self.state.current_cash:
            return None
        
        self.state.current_cash -= cost
        
        position = {
            'ticker': rec.ticker,
            'shares': rec.target_shares,
            'entry_price': rec.current_price,
            'entry_date': datetime.now().strftime('%Y-%m-%d'),
            'stop_loss': rec.stop_loss,
            'take_profit': rec.take_profit,
            'current_price': rec.current_price,
            'unrealized_pnl': 0,
            'unrealized_pnl_pct': 0,
        }
        self.state.positions.append(position)
        
        trade = {
            'ticker': rec.ticker,
            'action': 'BUY',
            'shares': rec.target_shares,
            'price': rec.current_price,
            'timestamp': datetime.now().isoformat(),
            'reason': rec.reason,
        }
        self.state.trade_history.append(trade)
        self.state.total_trades += 1
        
        logger.info(f"BUY {rec.target_shares} {rec.ticker} @ ${rec.current_price:.2f}")
        return trade
    
    def _execute_sell(self, rec: TradeRecommendation) -> Optional[Dict]:
        """Execute a sell trade."""
        # Find position
        pos_idx = None
        for i, p in enumerate(self.state.positions):
            if p['ticker'] == rec.ticker:
                pos_idx = i
                break
        
        if pos_idx is None:
            return None
        
        pos = self.state.positions.pop(pos_idx)
        proceeds = rec.current_price * pos['shares'] * 0.999  # 0.1% commission
        pnl = (rec.current_price - pos['entry_price']) * pos['shares']
        
        self.state.current_cash += proceeds
        
        trade = {
            'ticker': rec.ticker,
            'action': 'SELL',
            'shares': pos['shares'],
            'price': rec.current_price,
            'timestamp': datetime.now().isoformat(),
            'pnl': pnl,
            'reason': rec.reason,
        }
        self.state.trade_history.append(trade)
        self.state.total_trades += 1
        
        if pnl > 0:
            self.state.winning_trades += 1
        else:
            self.state.losing_trades += 1
        
        logger.info(f"SELL {pos['shares']} {rec.ticker} @ ${rec.current_price:.2f} (P&L: ${pnl:+.2f})")
        return trade
    
    def get_status(self) -> Dict:
        """Get current portfolio status."""
        if not self.state:
            self.load_state()
        
        if not self.state:
            return {'error': 'No state found. Initialize first.'}
        
        days_active = (datetime.now() - datetime.fromisoformat(self.state.initialized_at)).days
        
        return {
            'initialized_at': self.state.initialized_at,
            'days_active': days_active,
            'initial_capital': self.state.initial_capital,
            'current_cash': self.state.current_cash,
            'positions_value': self.state.positions_value,
            'total_value': self.state.total_value,
            'total_return': self.state.total_return,
            'total_trades': self.state.total_trades,
            'winning_trades': self.state.winning_trades,
            'losing_trades': self.state.losing_trades,
            'win_rate': self.state.winning_trades / self.state.total_trades if self.state.total_trades > 0 else 0,
            'positions': self.state.positions,
            'last_run': self.state.last_run,
        }
    
    def format_status(self) -> str:
        """Format status for display."""
        status = self.get_status()
        
        if 'error' in status:
            return status['error']
        
        lines = [
            "",
            "=" * 70,
            "📊 LONG RUN SIMULATION STATUS",
            "=" * 70,
            f"Started: {status['initialized_at'][:10]} ({status['days_active']} days active)",
            f"Last Run: {status['last_run'][:19] if status['last_run'] else 'Never'}",
            "",
            "-" * 70,
            "💰 PORTFOLIO",
            "-" * 70,
            f"Initial Capital:   ${status['initial_capital']:>12,.2f}",
            f"Current Cash:      ${status['current_cash']:>12,.2f}",
            f"Positions Value:   ${status['positions_value']:>12,.2f}",
            f"Total Value:       ${status['total_value']:>12,.2f}",
            f"Total Return:      {status['total_return']:>12.2%}",
            "",
            "-" * 70,
            "📋 TRADES",
            "-" * 70,
            f"Total Trades:      {status['total_trades']:>12}",
            f"Winning:           {status['winning_trades']:>12}",
            f"Losing:            {status['losing_trades']:>12}",
            f"Win Rate:          {status['win_rate']:>12.1%}",
        ]
        
        if status['positions']:
            lines.extend([
                "",
                "-" * 70,
                "📈 OPEN POSITIONS",
                "-" * 70,
                f"{'Ticker':<8} {'Shares':>8} {'Entry':>10} {'Current':>10} {'P&L':>12} {'P&L%':>8}",
            ])
            for pos in status['positions']:
                pnl = pos.get('unrealized_pnl', 0)
                pnl_pct = pos.get('unrealized_pnl_pct', 0)
                lines.append(
                    f"{pos['ticker']:<8} {pos['shares']:>8} "
                    f"${pos['entry_price']:>9.2f} ${pos.get('current_price', 0):>9.2f} "
                    f"${pnl:>+11.2f} {pnl_pct:>+7.1f}%"
                )
        
        lines.append("=" * 70)
        return "\n".join(lines)
