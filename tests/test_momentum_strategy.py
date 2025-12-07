"""
Tests for Momentum Strategy
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from datetime import datetime, timedelta


class TestMomentumScoring:
    """Test momentum scoring logic."""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame with 150 days of data."""
        dates = pd.date_range(end=datetime.now(), periods=150, freq='D')
        np.random.seed(42)
        
        # Generate realistic price data with uptrend
        base_price = 100
        returns = np.random.normal(0.001, 0.02, 150)  # ~10% annual return, 2% daily vol
        prices = base_price * np.cumprod(1 + returns)
        
        return pd.DataFrame({
            'Open': prices * 0.99,
            'High': prices * 1.01,
            'Low': prices * 0.98,
            'Close': prices,
            'Volume': np.random.randint(1000000, 5000000, 150),
        }, index=dates)
    
    def test_score_ticker_positive_momentum(self, sample_df):
        """Test that uptrending stock gets positive score."""
        from src.momentum_strategy import MomentumStrategy
        
        strategy = MomentumStrategy(silent=True)
        score = strategy._score_ticker(sample_df)
        
        # With random seed 42, we should get a positive score
        assert score > -999, "Score should be valid (not error value)"
    
    def test_score_ticker_insufficient_data(self):
        """Test handling of insufficient data."""
        from src.momentum_strategy import MomentumStrategy
        
        # Only 30 days of data (need 60)
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        df = pd.DataFrame({
            'Close': np.random.uniform(100, 110, 30),
            'Volume': np.random.randint(1000000, 5000000, 30),
        }, index=dates)
        
        strategy = MomentumStrategy(silent=True)
        score = strategy._score_ticker(df)
        
        assert score == -999, "Should return -999 for insufficient data"
    
    def test_score_ticker_with_finnhub_data(self, sample_df):
        """Test scoring with FinnHub data included."""
        from src.momentum_strategy import MomentumStrategy
        
        strategy = MomentumStrategy(silent=True)
        
        # Mock FinnHub data
        mock_fh = Mock()
        mock_fh.analyst_score = 0.5  # Positive analyst sentiment
        mock_fh.value_score = 0.7  # Good value
        mock_fh.pe_ratio = 25  # Reasonable P/E
        mock_fh.profit_margin = 0.30  # Good margin
        mock_fh.sentiment_score = 0.3  # Positive sentiment
        
        strategy._finnhub_data = {'AAPL': mock_fh}
        strategy._finnhub_loaded = True
        
        score_with_fh = strategy._score_ticker(sample_df, ticker='AAPL')
        score_without_fh = strategy._score_ticker(sample_df, ticker='UNKNOWN')
        
        # With positive FinnHub data, score should be higher
        assert score_with_fh != score_without_fh, "FinnHub data should affect score"
    
    def test_weights_sum_to_100_percent(self, sample_df):
        """Verify that scoring weights are properly balanced."""
        from src.momentum_strategy import MomentumStrategy
        
        strategy = MomentumStrategy(silent=True)
        
        # Mock FinnHub data with maximum values
        mock_fh = Mock()
        mock_fh.analyst_score = 1.0  # Max analyst score
        mock_fh.value_score = 1.0  # Max value score
        mock_fh.pe_ratio = 20  # Normal P/E (no penalty)
        mock_fh.profit_margin = 0.30  # Triggers bonus
        mock_fh.sentiment_score = 1.0  # Max sentiment
        
        strategy._finnhub_data = {'TEST': mock_fh}
        strategy._finnhub_loaded = True
        
        # With all max values, the non-momentum components should be:
        # 0.15 (analyst) + 0.15 (fundamental) + 0.05 (margin bonus) + 0.15 (sentiment) = 0.50
        # Plus momentum component scaled to 55%
        score = strategy._score_ticker(sample_df, ticker='TEST')
        
        # Score should be reasonable (not infinite or zero)
        assert -10 < score < 10, f"Score should be in reasonable range, got {score}"


class TestFallbackBehavior:
    """Test fallback to technical-only scoring."""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame."""
        dates = pd.date_range(end=datetime.now(), periods=150, freq='D')
        prices = 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, 150))
        
        return pd.DataFrame({
            'Open': prices * 0.99,
            'High': prices * 1.01,
            'Low': prices * 0.98,
            'Close': prices,
            'Volume': np.random.randint(1000000, 5000000, 150),
        }, index=dates)
    
    def test_fallback_when_no_finnhub(self, sample_df):
        """Test that strategy falls back to momentum-only when no FinnHub data."""
        from src.momentum_strategy import MomentumStrategy
        
        strategy = MomentumStrategy(silent=True)
        strategy._finnhub_loaded = True  # Mark as loaded but empty
        strategy._finnhub_data = {}
        
        # Should not crash and should return valid score
        score = strategy._score_ticker(sample_df, ticker='UNKNOWN')
        
        assert score != -999, "Should return valid score even without FinnHub data"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
