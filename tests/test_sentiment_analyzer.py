"""
Tests for Sentiment Analyzer
"""

import pytest
from src.sentiment_analyzer import SentimentAnalyzer, SentimentResult, get_sentiment_analyzer


class TestSentimentAnalyzer:
    """Tests for SentimentAnalyzer class."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return SentimentAnalyzer()
    
    def test_positive_headline(self, analyzer):
        """Test bullish headline detection."""
        text = "NVDA beats earnings expectations, stock surges to record high"
        score = analyzer.analyze_text(text)
        assert score > 0.3, f"Expected positive score, got {score}"
    
    def test_negative_headline(self, analyzer):
        """Test bearish headline detection."""
        text = "Tesla recalls 500,000 vehicles, stock plunges on disappointing sales"
        score = analyzer.analyze_text(text)
        assert score < -0.3, f"Expected negative score, got {score}"
    
    def test_neutral_headline(self, analyzer):
        """Test neutral headline detection."""
        text = "Apple to hold annual shareholder meeting next week"
        score = analyzer.analyze_text(text)
        assert -0.3 <= score <= 0.3, f"Expected neutral score, got {score}"
    
    def test_financial_lexicon_upgrade(self, analyzer):
        """Test that 'upgrade' is recognized as positive."""
        text = "Goldman Sachs upgrades AAPL to buy"
        score = analyzer.analyze_text(text)
        assert score > 0.2, f"Expected positive score for upgrade, got {score}"
    
    def test_financial_lexicon_downgrade(self, analyzer):
        """Test that 'downgrade' is recognized as negative."""
        text = "Morgan Stanley downgrades TSLA to sell"
        score = analyzer.analyze_text(text)
        assert score < -0.2, f"Expected negative score for downgrade, got {score}"
    
    def test_analyze_headlines_average(self, analyzer):
        """Test averaging multiple headlines."""
        headlines = [
            "Stock surges on strong earnings",  # Positive
            "Company announces new partnership",  # Slightly positive
            "Quarterly revenue meets expectations",  # Neutral
        ]
        score = analyzer.analyze_headlines(headlines)
        assert score > 0, f"Expected positive average, got {score}"
    
    def test_analyze_empty_headlines(self, analyzer):
        """Test empty headlines list."""
        score = analyzer.analyze_headlines([])
        assert score == 0.0
    
    def test_analyze_news_with_decay(self, analyzer):
        """Test news analysis with time decay."""
        import time
        now = time.time()
        
        news = [
            {"headline": "Great earnings beat!", "summary": "Strong quarter", "datetime": now},
            {"headline": "Positive outlook", "summary": "", "datetime": now - 86400},  # 1 day ago
        ]
        
        result = analyzer.analyze_news(news)
        
        assert isinstance(result, SentimentResult)
        assert result.news_count == 2
        assert result.score > 0  # Should be positive
        assert result.confidence > 0  # Should have some confidence
    
    def test_sentiment_result_counts(self, analyzer):
        """Test bullish/bearish/neutral counts."""
        import time
        now = time.time()
        
        news = [
            {"headline": "Stock surges 20% on earnings beat!", "datetime": now},  # Bullish
            {"headline": "Company misses targets badly", "datetime": now},  # Bearish
            {"headline": "Meeting scheduled", "datetime": now},  # Neutral
        ]
        
        result = analyzer.analyze_news(news)
        
        # At least one should be bullish (surge + beat)
        assert result.bullish_count >= 1 or result.score > 0, f"Expected positive sentiment, got {result}"
        assert result.bearish_count >= 1
    
    def test_singleton_instance(self):
        """Test singleton pattern."""
        analyzer1 = get_sentiment_analyzer()
        analyzer2 = get_sentiment_analyzer()
        assert analyzer1 is analyzer2


class TestFinancialLexicon:
    """Test financial-specific vocabulary."""
    
    @pytest.fixture
    def analyzer(self):
        return SentimentAnalyzer()
    
    @pytest.mark.parametrize("word,expected_sign", [
        ("beats", 1),
        ("outperform", 1),
        ("bullish", 1),
        ("surge", 1),
        ("rally", 1),
        ("misses", -1),
        ("underperform", -1),
        ("bearish", -1),
        ("crash", -1),
        ("selloff", -1),
    ])
    def test_financial_words(self, analyzer, word, expected_sign):
        """Test individual financial words are scored correctly."""
        text = f"Stock {word} today"
        score = analyzer.analyze_text(text)
        
        if expected_sign > 0:
            assert score > 0, f"Expected positive for '{word}', got {score}"
        else:
            assert score < 0, f"Expected negative for '{word}', got {score}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
