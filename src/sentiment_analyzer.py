"""
Sentiment Analyzer Module
Analyzes financial news sentiment using VADER.
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    """Result from sentiment analysis."""
    ticker: str
    score: float  # -1 (bearish) to +1 (bullish)
    confidence: float  # 0-1, based on news count and recency
    news_count: int
    bullish_count: int
    bearish_count: int
    neutral_count: int
    avg_headline_score: float
    latest_headline: Optional[str] = None
    

class SentimentAnalyzer:
    """
    Analyze financial news sentiment using VADER.
    
    VADER (Valence Aware Dictionary and sEntiment Reasoner) is specifically
    tuned for social media and news, making it suitable for financial headlines.
    """
    
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        
        # Add financial-specific words to VADER lexicon
        self._add_financial_lexicon()
    
    def _add_financial_lexicon(self):
        """Add finance-specific words to improve accuracy."""
        financial_words = {
            # Positive
            'beat': 2.0,
            'beats': 2.0,
            'outperform': 2.5,
            'upgrade': 2.0,
            'upgrades': 2.0,
            'bullish': 2.5,
            'rally': 2.0,
            'surge': 2.5,
            'soar': 2.5,
            'breakout': 2.0,
            'record high': 3.0,
            'all-time high': 3.0,
            'strong buy': 3.0,
            'overweight': 1.5,
            'outperforming': 2.0,
            'exceeds': 1.5,
            'exceeded': 1.5,
            'growth': 1.5,
            'profit': 1.5,
            'profitable': 1.5,
            
            # Negative
            'miss': -2.0,
            'misses': -2.0,
            'underperform': -2.5,
            'downgrade': -2.0,
            'downgrades': -2.0,
            'bearish': -2.5,
            'selloff': -2.5,
            'sell-off': -2.5,
            'plunge': -2.5,
            'crash': -3.0,
            'collapse': -3.0,
            'record low': -3.0,
            'strong sell': -3.0,
            'underweight': -1.5,
            'disappoints': -2.0,
            'disappointing': -2.0,
            'loss': -1.5,
            'losses': -1.5,
            'decline': -1.5,
            'warning': -2.0,
            'cautious': -1.0,
            'uncertainty': -1.5,
            'volatile': -1.0,
            'recession': -2.5,
            'layoffs': -2.0,
            'layoff': -2.0,
        }
        
        self.analyzer.lexicon.update(financial_words)
    
    def analyze_text(self, text: str) -> float:
        """
        Analyze sentiment of a single text.
        
        Args:
            text: Text to analyze (headline or summary)
            
        Returns:
            Compound score from -1 (very negative) to +1 (very positive)
        """
        if not text:
            return 0.0
        
        scores = self.analyzer.polarity_scores(text)
        return scores['compound']
    
    def analyze_headlines(self, headlines: List[str]) -> float:
        """
        Analyze sentiment of multiple headlines.
        
        Args:
            headlines: List of news headlines
            
        Returns:
            Average compound score
        """
        if not headlines:
            return 0.0
        
        scores = [self.analyze_text(h) for h in headlines]
        return sum(scores) / len(scores)
    
    def analyze_news(
        self, 
        news: List[Dict],
        decay_hours: float = 72.0
    ) -> SentimentResult:
        """
        Analyze sentiment from news items with time decay.
        
        More recent news is weighted more heavily.
        
        Args:
            news: List of news items with 'headline', 'summary', 'datetime'
            decay_hours: Hours after which weight drops to 50%
            
        Returns:
            SentimentResult with aggregated sentiment
        """
        if not news:
            return SentimentResult(
                ticker="",
                score=0.0,
                confidence=0.0,
                news_count=0,
                bullish_count=0,
                bearish_count=0,
                neutral_count=0,
                avg_headline_score=0.0,
            )
        
        now = datetime.now()
        weighted_scores = []
        bullish = 0
        bearish = 0
        neutral = 0
        
        for item in news:
            headline = item.get('headline', '')
            summary = item.get('summary', '')
            
            # Combine headline (weighted more) and summary
            headline_score = self.analyze_text(headline)
            summary_score = self.analyze_text(summary) if summary else 0
            
            # Headline is more important (70% weight)
            combined_score = headline_score * 0.7 + summary_score * 0.3
            
            # Time decay: more recent = higher weight
            news_time = item.get('datetime')
            if news_time:
                try:
                    if isinstance(news_time, (int, float)):
                        news_dt = datetime.fromtimestamp(news_time)
                    else:
                        news_dt = datetime.fromisoformat(str(news_time).replace('Z', '+00:00'))
                    
                    hours_ago = (now - news_dt.replace(tzinfo=None)).total_seconds() / 3600
                    # Exponential decay: weight = 0.5^(hours/decay_hours)
                    weight = 0.5 ** (hours_ago / decay_hours)
                except:
                    weight = 0.5  # Default weight if parsing fails
            else:
                weight = 0.5
            
            weighted_scores.append(combined_score * weight)
            
            # Count sentiment categories
            if combined_score > 0.1:
                bullish += 1
            elif combined_score < -0.1:
                bearish += 1
            else:
                neutral += 1
        
        # Calculate final score
        if weighted_scores:
            total_weight = sum(abs(s) for s in weighted_scores) or 1
            final_score = sum(weighted_scores) / len(weighted_scores)
        else:
            final_score = 0.0
        
        # Confidence based on news count and agreement
        news_count = len(news)
        agreement = max(bullish, bearish, neutral) / news_count if news_count > 0 else 0
        
        # More news + higher agreement = higher confidence
        confidence = min(1.0, (news_count / 10) * 0.5 + agreement * 0.5)
        
        return SentimentResult(
            ticker=news[0].get('related', '') if news else "",
            score=max(-1.0, min(1.0, final_score)),  # Clamp to [-1, 1]
            confidence=confidence,
            news_count=news_count,
            bullish_count=bullish,
            bearish_count=bearish,
            neutral_count=neutral,
            avg_headline_score=sum(self.analyze_text(n.get('headline', '')) for n in news) / news_count if news_count > 0 else 0,
            latest_headline=news[0].get('headline') if news else None,
        )
    
    def analyze_ticker(self, ticker: str, news: List[Dict]) -> SentimentResult:
        """
        Analyze sentiment for a specific ticker.
        
        Args:
            ticker: Stock ticker symbol
            news: List of news items
            
        Returns:
            SentimentResult for the ticker
        """
        result = self.analyze_news(news)
        result.ticker = ticker
        return result


# Singleton instance
_analyzer: Optional[SentimentAnalyzer] = None


def get_sentiment_analyzer() -> SentimentAnalyzer:
    """Get singleton SentimentAnalyzer instance."""
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentAnalyzer()
    return _analyzer


if __name__ == "__main__":
    # Test the analyzer
    analyzer = SentimentAnalyzer()
    
    # Test headlines
    test_headlines = [
        "NVDA beats earnings expectations, stock surges 10%",
        "Apple iPhone sales disappoint in China",
        "Microsoft announces new AI partnership",
        "Tesla recalls 500,000 vehicles over safety concerns",
        "Amazon stock hits all-time high after strong quarter",
    ]
    
    print("Testing Sentiment Analysis:\n")
    for headline in test_headlines:
        score = analyzer.analyze_text(headline)
        sentiment = "🟢 Bullish" if score > 0.1 else "🔴 Bearish" if score < -0.1 else "⚪ Neutral"
        print(f"{sentiment} ({score:+.2f}): {headline}")
