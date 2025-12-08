"""
Tests for Unified Scorer
"""

import pytest
from src.unified_scorer import (
    UnifiedScorer,
    UnifiedScore,
    SignalStrength,
    WEIGHTS,
    THRESHOLDS,
    calculate_unified_score,
)


class TestSignalStrength:
    """Tests for SignalStrength enum."""
    
    def test_signal_values(self):
        """Test signal strength values."""
        assert SignalStrength.STRONG_BUY.value == 2
        assert SignalStrength.BUY.value == 1
        assert SignalStrength.HOLD.value == 0
        assert SignalStrength.SELL.value == -1
        assert SignalStrength.STRONG_SELL.value == -2


class TestUnifiedScorer:
    """Tests for UnifiedScorer class."""
    
    @pytest.fixture
    def scorer(self):
        """Create a scorer instance."""
        return UnifiedScorer()
    
    def test_default_weights(self, scorer):
        """Test default weights are applied."""
        assert scorer.weights['technical'] == 0.40
        assert scorer.weights['ml_prediction'] == 0.20
        assert scorer.weights['fundamental'] == 0.20
        assert scorer.weights['sentiment'] == 0.15
        assert scorer.weights['momentum'] == 0.05
        
        # Weights should sum to 1.0
        total = sum(scorer.weights.values())
        assert abs(total - 1.0) < 0.01
    
    def test_custom_weights(self):
        """Test custom weights are normalized."""
        custom = {'technical': 0.5, 'ml_prediction': 0.5}
        scorer = UnifiedScorer(weights=custom)
        
        # Should be normalized
        assert abs(scorer.weights['technical'] - 0.5) < 0.01
    
    def test_calculate_score_empty(self, scorer):
        """Test scoring with no data."""
        result = scorer.calculate_score()
        
        assert isinstance(result, UnifiedScore)
        assert result.score == 0.0
        assert result.signal == SignalStrength.HOLD
    
    def test_calculate_score_technical_only(self, scorer):
        """Test scoring with only technical analysis."""
        technical = {
            'overall': {
                'signal': 'BUY',
                'signal_value': 1.5,
            },
            'price': {'change_percent': 2.0},
            'raw_indicators': {'Volume_Ratio': 1.2},
        }
        
        result = scorer.calculate_score(technical_analysis=technical)
        
        assert result.score > 0
        assert result.signal in [SignalStrength.BUY, SignalStrength.STRONG_BUY]
        assert 'technical' in result.components
    
    def test_calculate_score_bullish(self, scorer):
        """Test bullish signal scoring."""
        technical = {
            'overall': {'signal': 'STRONG_BUY', 'signal_value': 2.0},
            'price': {'change_percent': 3.0},
        }
        
        result = scorer.calculate_score(technical_analysis=technical)
        
        assert result.score > 1.0
        assert result.signal in [SignalStrength.BUY, SignalStrength.STRONG_BUY]
    
    def test_calculate_score_bearish(self, scorer):
        """Test bearish signal scoring."""
        technical = {
            'overall': {'signal': 'STRONG_SELL', 'signal_value': -2.0},
            'price': {'change_percent': -3.0},
        }
        
        result = scorer.calculate_score(technical_analysis=technical)
        
        assert result.score < -1.0
        assert result.signal in [SignalStrength.SELL, SignalStrength.STRONG_SELL]
    
    def test_signal_classification(self, scorer):
        """Test signal classification thresholds."""
        # Strong buy
        assert scorer._classify_signal(1.5) == SignalStrength.STRONG_BUY
        # Buy
        assert scorer._classify_signal(0.7) == SignalStrength.BUY
        # Hold
        assert scorer._classify_signal(0.0) == SignalStrength.HOLD
        # Sell 
        assert scorer._classify_signal(-0.7) == SignalStrength.SELL
        # Strong sell
        assert scorer._classify_signal(-1.5) == SignalStrength.STRONG_SELL
    
    def test_confidence_calculation(self, scorer):
        """Test confidence calculation."""
        # High confidence: all signals agree
        components = {'technical': 1.5, 'ml': 1.2, 'fundamental': 1.0}
        confidence = scorer._calculate_confidence(components)
        assert confidence > 0.5
        
        # Low confidence: signals disagree
        components = {'technical': 1.5, 'ml': -1.0, 'fundamental': 0.0}
        confidence = scorer._calculate_confidence(components)
        assert confidence < 0.7
    
    def test_get_signal_name(self, scorer):
        """Test signal name formatting."""
        result = UnifiedScore(
            score=1.5,
            signal=SignalStrength.STRONG_BUY,
            confidence=0.8,
            components={},
            reason="Test"
        )
        
        name = scorer.get_signal_name(result)
        assert name == "Strong Buy"
    
    def test_get_action_recommendation(self, scorer):
        """Test action recommendation generation."""
        result = UnifiedScore(
            score=1.5,
            signal=SignalStrength.STRONG_BUY,
            confidence=0.8,
            components={},
            reason="Test"
        )
        
        action = scorer.get_action_recommendation(result)
        assert "STRONG BUY" in action
    
    def test_include_finnhub_false(self):
        """Test scoring without Finnhub data."""
        scorer = UnifiedScorer(include_finnhub=False)
        
        # Even with finnhub_data, it should be ignored
        class MockFinnhub:
            analyst_score = 1.0
            value_score = 0.8
            sentiment_score = 0.5
        
        result = scorer.calculate_score(finnhub_data=MockFinnhub())
        
        # Should not include fundamental or sentiment
        assert 'fundamental' not in result.components
        assert 'sentiment' not in result.components


class TestConvenienceFunction:
    """Tests for calculate_unified_score convenience function."""
    
    def test_convenience_function(self):
        """Test the convenience function."""
        technical = {
            'overall': {'signal': 'BUY', 'signal_value': 1.0},
        }
        
        result = calculate_unified_score(technical_analysis=technical)
        
        assert isinstance(result, UnifiedScore)
        assert result.score != 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
