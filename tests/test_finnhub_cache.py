"""
Tests for Finnhub Client Cache
"""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from src.finnhub_client import (
    FinnHubClient,
    TickerData,
    CACHE_TTL_HOURS,
)


class TestFinnhubCache:
    """Tests for Finnhub cache functionality."""
    
    @pytest.fixture
    def temp_cache_file(self):
        """Create a temporary cache file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            yield f.name
        Path(f.name).unlink(missing_ok=True)
    
    def test_cache_ttl(self):
        """Test cache TTL is 2 hours."""
        assert CACHE_TTL_HOURS == 2
    
    @patch('src.finnhub_client.FINNHUB_API_KEY', 'test_key')
    def test_load_empty_cache(self, temp_cache_file):
        """Test loading when cache file doesn't exist."""
        # Remove file if exists
        Path(temp_cache_file).unlink(missing_ok=True)
        
        client = FinnHubClient(cache_file=temp_cache_file)
        assert len(client._cache) == 0
    
    @patch('src.finnhub_client.FINNHUB_API_KEY', 'test_key')
    def test_load_valid_cache(self, temp_cache_file):
        """Test loading valid cache within TTL."""
        # Create valid cache file
        cache_data = {
            'timestamp': datetime.now().isoformat(),
            'tickers': {
                'AAPL': {
                    'ticker': 'AAPL',
                    'current_price': 150.0,
                    'change': 2.5,
                    'change_percent': 1.5,
                    'high': None,
                    'low': None,
                    'open': None,
                    'previous_close': None,
                    'pe_ratio': 25.0,
                    'pe_forward': None,
                    'pb_ratio': None,
                    'ps_ratio': None,
                    'eps': None,
                    'eps_growth': None,
                    'revenue_growth': None,
                    'profit_margin': 0.25,
                    'gross_margin': None,
                    'roe': 0.15,
                    'roa': None,
                    'debt_equity': None,
                    'current_ratio': None,
                    'dividend_yield': None,
                    'beta': None,
                    'market_cap': 2500000000000,
                    'week52_high': None,
                    'week52_low': None,
                    'news': [],
                    'analyst_buy': 30,
                    'analyst_hold': 10,
                    'analyst_sell': 2,
                    'analyst_strong_buy': 15,
                    'analyst_strong_sell': 0,
                }
            }
        }
        
        with open(temp_cache_file, 'w') as f:
            json.dump(cache_data, f)
        
        client = FinnHubClient(cache_file=temp_cache_file)
        
        assert 'AAPL' in client._cache
        assert client._cache['AAPL'].current_price == 150.0
        assert client.is_cache_valid()
    
    @patch('src.finnhub_client.FINNHUB_API_KEY', 'test_key')
    def test_load_expired_cache(self, temp_cache_file):
        """Test loading expired cache (beyond TTL)."""
        # Create expired cache file
        expired_time = datetime.now() - timedelta(hours=CACHE_TTL_HOURS + 1)
        cache_data = {
            'timestamp': expired_time.isoformat(),
            'tickers': {
                'AAPL': {
                    'ticker': 'AAPL',
                    'current_price': 150.0,
                    'change': None,
                    'change_percent': None,
                    'high': None,
                    'low': None,
                    'open': None,
                    'previous_close': None,
                    'pe_ratio': None,
                    'pe_forward': None,
                    'pb_ratio': None,
                    'ps_ratio': None,
                    'eps': None,
                    'eps_growth': None,
                    'revenue_growth': None,
                    'profit_margin': None,
                    'gross_margin': None,
                    'roe': None,
                    'roa': None,
                    'debt_equity': None,
                    'current_ratio': None,
                    'dividend_yield': None,
                    'beta': None,
                    'market_cap': None,
                    'week52_high': None,
                    'week52_low': None,
                    'news': [],
                    'analyst_buy': 0,
                    'analyst_hold': 0,
                    'analyst_sell': 0,
                    'analyst_strong_buy': 0,
                    'analyst_strong_sell': 0,
                }
            }
        }
        
        with open(temp_cache_file, 'w') as f:
            json.dump(cache_data, f)
        
        client = FinnHubClient(cache_file=temp_cache_file)
        
        # Cache should be empty due to expiration
        assert len(client._cache) == 0
        assert not client.is_cache_valid()
    
    @patch('src.finnhub_client.FINNHUB_API_KEY', 'test_key')
    def test_save_cache(self, temp_cache_file):
        """Test saving cache to file."""
        client = FinnHubClient(cache_file=temp_cache_file)
        client._cache = {
            'MSFT': TickerData(
                ticker='MSFT',
                current_price=400.0,
                analyst_buy=20,
            )
        }
        
        success = client._save_cache()
        
        assert success
        assert Path(temp_cache_file).exists()
        
        # Verify file contents
        with open(temp_cache_file) as f:
            data = json.load(f)
        
        assert 'timestamp' in data
        assert 'MSFT' in data['tickers']
        assert data['tickers']['MSFT']['current_price'] == 400.0
    
    @patch('src.finnhub_client.FINNHUB_API_KEY', 'test_key')
    def test_invalidate_cache(self, temp_cache_file):
        """Test cache invalidation."""
        # Create cache file
        with open(temp_cache_file, 'w') as f:
            json.dump({'timestamp': datetime.now().isoformat(), 'tickers': {}}, f)
        
        client = FinnHubClient(cache_file=temp_cache_file)
        client._cache = {'AAPL': TickerData(ticker='AAPL')}
        client._cache_timestamp = datetime.now()
        
        client.invalidate_cache()
        
        assert len(client._cache) == 0
        assert client._cache_timestamp is None
        assert not Path(temp_cache_file).exists()
    
    @patch('src.finnhub_client.FINNHUB_API_KEY', 'test_key')
    def test_is_available(self, temp_cache_file):
        """Test is_available property."""
        client = FinnHubClient(cache_file=temp_cache_file)
        
        # No cache
        assert not client.is_available
        
        # With cache
        client._cache = {'AAPL': TickerData(ticker='AAPL')}
        assert client.is_available


class TestTickerData:
    """Tests for TickerData class."""
    
    def test_analyst_score(self):
        """Test analyst score calculation."""
        data = TickerData(
            ticker='TEST',
            analyst_buy=10,
            analyst_hold=5,
            analyst_sell=2,
            analyst_strong_buy=8,
            analyst_strong_sell=0,
        )
        
        # Score should be positive (more buys than sells)
        assert data.analyst_score > 0
    
    def test_analyst_score_empty(self):
        """Test analyst score with no ratings."""
        data = TickerData(ticker='TEST')
        assert data.analyst_score == 0
    
    def test_value_score(self):
        """Test value score calculation."""
        data = TickerData(
            ticker='TEST',
            pe_ratio=15.0,  # Good (< 25)
            profit_margin=0.25,  # Good (> 20%)
            roe=0.18,  # Good (> 10%)
        )
        
        # Should be positive with good fundamentals
        assert data.value_score > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
