"""
Tests for Technical Indicators
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime


class TestTechnicalIndicators:
    """Tests for TechnicalIndicators class."""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample OHLCV DataFrame with 250 days."""
        dates = pd.date_range(end=datetime.now(), periods=250, freq='D')
        np.random.seed(42)
        
        base_price = 100
        returns = np.random.normal(0.0005, 0.015, 250)
        close = base_price * np.cumprod(1 + returns)
        
        return pd.DataFrame({
            'Open': close * (1 + np.random.uniform(-0.01, 0.01, 250)),
            'High': close * (1 + np.abs(np.random.uniform(0, 0.02, 250))),
            'Low': close * (1 - np.abs(np.random.uniform(0, 0.02, 250))),
            'Close': close,
            'Volume': np.random.randint(1000000, 10000000, 250),
        }, index=dates)
    
    def test_indicator_calculation(self, sample_df):
        """Test that all indicators are calculated."""
        from src.indicators import TechnicalIndicators
        
        ti = TechnicalIndicators(sample_df)
        df = ti.get_dataframe()
        
        # Check key indicators exist
        expected_columns = ['RSI', 'SMA_20', 'SMA_50', 'SMA_200', 'MACD', 'ATR', 'ADX']
        for col in expected_columns:
            assert col in df.columns, f"Missing indicator: {col}"
    
    def test_rsi_bounds(self, sample_df):
        """Test RSI is within valid bounds (0-100)."""
        from src.indicators import TechnicalIndicators
        
        ti = TechnicalIndicators(sample_df)
        df = ti.get_dataframe()
        
        rsi = df['RSI'].dropna()
        assert rsi.min() >= 0, f"RSI below 0: {rsi.min()}"
        assert rsi.max() <= 100, f"RSI above 100: {rsi.max()}"
    
    def test_comprehensive_analysis(self, sample_df):
        """Test comprehensive analysis returns expected structure."""
        from src.indicators import TechnicalIndicators
        
        ti = TechnicalIndicators(sample_df)
        analysis = ti.get_comprehensive_analysis()
        
        # Check structure
        assert 'price' in analysis
        assert 'indicators' in analysis
        assert 'overall' in analysis
        
        # Check price info
        assert 'current' in analysis['price']
        assert 'change' in analysis['price']
        assert 'change_percent' in analysis['price']
        
        # Check overall recommendation
        assert 'signal' in analysis['overall']
        assert 'signal_value' in analysis['overall']
        assert 'score' in analysis['overall']
        assert 'action' in analysis['overall']
    
    def test_signal_values(self, sample_df):
        """Test that signals are in valid range."""
        from src.indicators import TechnicalIndicators, Signal
        
        ti = TechnicalIndicators(sample_df)
        analysis = ti.get_comprehensive_analysis()
        
        # Signal values should be between -2 and 2
        for name, indicator in analysis['indicators'].items():
            assert -2 <= indicator['signal_value'] <= 2, \
                f"Invalid signal value for {name}: {indicator['signal_value']}"
    
    def test_latest_indicators(self, sample_df):
        """Test getting latest indicator values."""
        from src.indicators import TechnicalIndicators
        
        ti = TechnicalIndicators(sample_df)
        latest = ti.get_latest_indicators()
        
        # Should be a dict with numeric values
        assert isinstance(latest, dict)
        assert len(latest) > 0
        
        for key, value in latest.items():
            assert isinstance(value, (int, float)), \
                f"Non-numeric value for {key}: {value}"


class TestRSIAnalysis:
    """Test RSI signal generation."""
    
    @pytest.fixture
    def oversold_df(self):
        """Create DataFrame with oversold conditions (RSI < 30)."""
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        
        # Declining prices
        close = 100 * np.cumprod([0.98] * 100)  # Continuous decline
        
        return pd.DataFrame({
            'Open': close * 1.01,
            'High': close * 1.02,
            'Low': close * 0.99,
            'Close': close,
            'Volume': [1000000] * 100,
        }, index=dates)
    
    def test_oversold_detection(self, oversold_df):
        """Test that continuous decline triggers oversold RSI."""
        from src.indicators import TechnicalIndicators, Signal
        
        ti = TechnicalIndicators(oversold_df)
        rsi_result = ti.analyze_rsi()
        
        # With continuous decline, RSI should be low
        assert rsi_result.value < 50, f"Expected low RSI, got {rsi_result.value}"


class TestBollingerBands:
    """Test Bollinger Bands analysis."""
    
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame."""
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        close = 100 + np.random.normal(0, 2, 100).cumsum()
        
        return pd.DataFrame({
            'Open': close * 0.99,
            'High': close * 1.01,
            'Low': close * 0.98,
            'Close': close,
            'Volume': [1000000] * 100,
        }, index=dates)
    
    def test_bands_calculated(self, sample_df):
        """Test Bollinger Bands are calculated correctly."""
        from src.indicators import TechnicalIndicators
        
        ti = TechnicalIndicators(sample_df)
        df = ti.get_dataframe()
        
        # Check bands exist
        assert 'BB_Upper' in df.columns
        assert 'BB_Middle' in df.columns
        assert 'BB_Lower' in df.columns
        
        # Upper should be above lower
        latest = df.iloc[-1]
        if pd.notna(latest['BB_Upper']) and pd.notna(latest['BB_Lower']):
            assert latest['BB_Upper'] > latest['BB_Lower']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
