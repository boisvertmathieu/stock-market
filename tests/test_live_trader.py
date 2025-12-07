"""
Tests for Live Trader
"""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch

from src.live_trader import LiveTrader, PortfolioState, TradeAction, TradeRecommendation


class TestPortfolioState:
    """Tests for PortfolioState class."""
    
    def test_from_dict(self):
        """Test creating state from dictionary."""
        data = {
            'initialized_at': '2025-12-06T10:00:00',
            'initial_capital': 10000,
            'current_cash': 8000,
            'positions': [
                {'ticker': 'AAPL', 'shares': 10, 'entry_price': 200, 'current_price': 210}
            ],
            'trade_history': [],
            'last_run': '2025-12-06T12:00:00',
            'total_trades': 1,
            'winning_trades': 0,
            'losing_trades': 0,
        }
        
        state = PortfolioState.from_dict(data)
        
        assert state.initial_capital == 10000
        assert state.current_cash == 8000
        assert len(state.positions) == 1
        assert state.positions[0]['ticker'] == 'AAPL'
    
    def test_total_value(self):
        """Test total value calculation."""
        state = PortfolioState(
            initialized_at='2025-12-06T10:00:00',
            initial_capital=10000,
            current_cash=5000,
            positions=[
                {'ticker': 'AAPL', 'shares': 10, 'entry_price': 200, 'current_price': 250}
            ]
        )
        
        # Cash (5000) + Positions (10 * 250 = 2500)
        assert state.positions_value == 2500
        assert state.total_value == 7500
    
    def test_total_return(self):
        """Test return calculation."""
        state = PortfolioState(
            initialized_at='2025-12-06T10:00:00',
            initial_capital=10000,
            current_cash=11000,
            positions=[]
        )
        
        assert state.total_return == 0.10  # 10% return


class TestLiveTrader:
    """Tests for LiveTrader class."""
    
    @pytest.fixture
    def temp_state_file(self):
        """Create a temporary state file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            yield f.name
        Path(f.name).unlink(missing_ok=True)
    
    def test_initialize(self, temp_state_file):
        """Test initialization creates state file."""
        trader = LiveTrader(state_file=temp_state_file)
        state = trader.initialize(10000)
        
        assert state.initial_capital == 10000
        assert state.current_cash == 10000
        assert state.positions == []
        assert Path(temp_state_file).exists()
    
    def test_load_state(self, temp_state_file):
        """Test loading existing state."""
        # Create state file
        state_data = {
            'initialized_at': '2025-12-06T10:00:00',
            'initial_capital': 10000,
            'current_cash': 9000,
            'positions': [],
            'trade_history': [],
            'last_run': '',
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
        }
        Path(temp_state_file).parent.mkdir(parents=True, exist_ok=True)
        with open(temp_state_file, 'w') as f:
            json.dump(state_data, f)
        
        trader = LiveTrader(state_file=temp_state_file)
        state = trader.load_state()
        
        assert state is not None
        assert state.current_cash == 9000
    
    def test_load_state_not_found(self, temp_state_file):
        """Test loading non-existent state returns None."""
        trader = LiveTrader(state_file='/nonexistent/path.json')
        state = trader.load_state()
        
        assert state is None
    
    @patch('src.live_trader.DataFetcher')
    @patch('src.live_trader.get_finnhub_client')
    def test_calculate_signal_score(self, mock_finnhub, mock_fetcher, temp_state_file):
        """Test signal score calculation."""
        trader = LiveTrader(state_file=temp_state_file)
        
        # Test with technical signal only
        data = {
            'technical_signal': 1.5,
            'finnhub': None
        }
        score = trader._calculate_signal_score(data)
        assert score == 0.675  # 1.5 * 0.45 (45% weight for technical)
        
        # Test with FinnHub data including sentiment
        data_with_fh = {
            'technical_signal': 1.0,
            'finnhub': {
                'analyst_score': 0.5,
                'value_score': 0.6,
                'change_percent': 2.0,
                'sentiment_score': 0.5,
            }
        }
        score_fh = trader._calculate_signal_score(data_with_fh)
        # 0.45 + (0.5 * 0.20) + (0.6 * 0.10) + 0.05 + (0.5 * 0.20) = 0.45 + 0.10 + 0.06 + 0.05 + 0.10 = 0.76
        assert score_fh > score


class TestTradeRecommendation:
    """Tests for TradeRecommendation class."""
    
    def test_buy_recommendation(self):
        """Test creating buy recommendation."""
        rec = TradeRecommendation(
            ticker='AAPL',
            action=TradeAction.BUY,
            confidence=0.85,
            current_price=280.0,
            target_shares=10,
            reason='Strong momentum',
            stop_loss=260.0,
            take_profit=320.0,
        )
        
        assert rec.ticker == 'AAPL'
        assert rec.action == TradeAction.BUY
        assert rec.confidence == 0.85
        assert rec.stop_loss == 260.0
        assert rec.take_profit == 320.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
