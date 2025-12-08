"""
Unified Scorer Module
Centralized signal scoring with consistent weights across all analysis modules.
Combines technical, ML, fundamental, and sentiment signals into a unified score.
"""

import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SignalStrength(Enum):
    """Unified signal strength classification."""
    STRONG_BUY = 2
    BUY = 1
    HOLD = 0
    SELL = -1
    STRONG_SELL = -2


@dataclass
class UnifiedScore:
    """Result container for unified scoring."""
    score: float  # -2 to +2
    signal: SignalStrength
    confidence: float  # 0 to 1
    components: Dict[str, float]  # Breakdown of score components
    reason: str  # Human-readable explanation


# Unified weights (optimized for robust signal generation)
WEIGHTS = {
    'technical': 0.40,      # RSI, MACD, MA, ADX, Bollinger, Stochastic
    'ml_prediction': 0.20,  # XGBoost ML model predictions
    'fundamental': 0.20,    # P/E, ROE, profit margin, analyst score
    'sentiment': 0.15,      # News sentiment from Finnhub
    'momentum': 0.05,       # Short-term price momentum
}

# Thresholds for signal classification
THRESHOLDS = {
    'strong_buy': 1.2,
    'buy': 0.5,
    'sell': -0.5,
    'strong_sell': -1.2,
}


class UnifiedScorer:
    """
    Centralized scoring engine for market analysis.
    Provides consistent signal scoring across all application modules.
    """
    
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        include_finnhub: bool = True,
    ):
        """
        Initialize the scorer.
        
        Args:
            weights: Custom weights (uses defaults if None)
            include_finnhub: Whether to include Finnhub data in scoring
        """
        self.weights = weights or WEIGHTS.copy()
        self.include_finnhub = include_finnhub
        
        # Validate weights sum to 1.0
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.01:
            logger.warning(f"Weights sum to {total}, normalizing to 1.0")
            for key in self.weights:
                self.weights[key] /= total
    
    def calculate_score(
        self,
        technical_analysis: Optional[Dict[str, Any]] = None,
        ml_result: Optional[Any] = None,
        finnhub_data: Optional[Any] = None,
        price_data: Optional[Dict[str, float]] = None,
    ) -> UnifiedScore:
        """
        Calculate unified score from all available signals.
        
        Args:
            technical_analysis: Result from TechnicalIndicators.get_comprehensive_analysis()
            ml_result: Result from EnhancedMLPredictor.predict() or MLPredictor.predict()
            finnhub_data: TickerData from FinnHubClient
            price_data: Dict with 'current', 'change_percent', 'previous' keys
            
        Returns:
            UnifiedScore with overall score and breakdown
        """
        components = {}
        available_weight = 0.0
        weighted_score = 0.0
        reasons = []
        
        # 1. Technical Analysis Score (40%)
        if technical_analysis:
            tech_score = self._extract_technical_score(technical_analysis)
            components['technical'] = tech_score
            weighted_score += tech_score * self.weights['technical']
            available_weight += self.weights['technical']
            
            if abs(tech_score) >= 1.0:
                signal_name = technical_analysis.get('overall', {}).get('signal', 'Unknown')
                reasons.append(f"Tech: {signal_name}")
        
        # 2. ML Prediction Score (20%)
        if ml_result:
            ml_score = self._extract_ml_score(ml_result)
            components['ml_prediction'] = ml_score
            weighted_score += ml_score * self.weights['ml_prediction']
            available_weight += self.weights['ml_prediction']
            
            if abs(ml_score) >= 1.0:
                pred_name = getattr(getattr(ml_result, 'prediction', None), 'name', 'Unknown')
                reasons.append(f"ML: {pred_name}")
        
        # 3. Fundamental Score (20%) - from Finnhub
        if finnhub_data and self.include_finnhub:
            fund_score = self._extract_fundamental_score(finnhub_data)
            components['fundamental'] = fund_score
            weighted_score += fund_score * self.weights['fundamental']
            available_weight += self.weights['fundamental']
            
            if abs(fund_score) >= 0.5:
                reasons.append(f"Fund: {'positive' if fund_score > 0 else 'negative'}")
        
        # 4. Sentiment Score (15%) - from Finnhub news
        if finnhub_data and self.include_finnhub:
            sent_score = self._extract_sentiment_score(finnhub_data)
            components['sentiment'] = sent_score
            weighted_score += sent_score * self.weights['sentiment']
            available_weight += self.weights['sentiment']
            
            if abs(sent_score) >= 0.5:
                reasons.append(f"Sent: {'bullish' if sent_score > 0 else 'bearish'}")
        
        # 5. Momentum Score (5%)
        if price_data or (technical_analysis and 'raw_indicators' in technical_analysis):
            mom_score = self._extract_momentum_score(price_data, technical_analysis)
            components['momentum'] = mom_score
            weighted_score += mom_score * self.weights['momentum']
            available_weight += self.weights['momentum']
        
        # Normalize score if not all components available
        if available_weight > 0 and available_weight < 1.0:
            weighted_score = weighted_score / available_weight
            # Mark that we're using partial data
            components['_partial'] = True
        
        # Clamp to -2 to +2 range
        final_score = max(-2.0, min(2.0, weighted_score))
        
        # Determine signal strength
        signal = self._classify_signal(final_score)
        
        # Calculate confidence based on agreement of components
        confidence = self._calculate_confidence(components)
        
        # Build reason string
        if not reasons:
            reasons.append("Mixed signals")
        reason_str = " | ".join(reasons)
        
        return UnifiedScore(
            score=round(final_score, 3),
            signal=signal,
            confidence=round(confidence, 2),
            components=components,
            reason=reason_str,
        )
    
    def _extract_technical_score(self, analysis: Dict) -> float:
        """Extract normalized score from technical analysis."""
        try:
            # Get overall signal value (-2 to +2)
            return float(analysis.get('overall', {}).get('signal_value', 0))
        except (TypeError, ValueError):
            return 0.0
    
    def _extract_ml_score(self, ml_result: Any) -> float:
        """Extract normalized score from ML prediction."""
        try:
            if ml_result is None:
                return 0.0
            
            # Handle EnhancedMLPredictor result
            if hasattr(ml_result, 'prediction'):
                pred = ml_result.prediction
                if hasattr(pred, 'value'):
                    return float(pred.value)
            
            # Handle legacy predictor result
            if isinstance(ml_result, dict):
                return float(ml_result.get('prediction_value', 0))
            
            return 0.0
        except (TypeError, ValueError):
            return 0.0
    
    def _extract_fundamental_score(self, finnhub_data: Any) -> float:
        """Extract normalized score from fundamental data."""
        try:
            if finnhub_data is None:
                return 0.0
            
            score = 0.0
            
            # Value score (0 to 1 range, from TickerData.value_score property)
            if hasattr(finnhub_data, 'value_score'):
                value = finnhub_data.value_score
                score += value * 0.5  # Contribute up to 0.5
            
            # Analyst score (-1 to +1 range)
            if hasattr(finnhub_data, 'analyst_score'):
                analyst = finnhub_data.analyst_score
                score += analyst  # Contribute up to 1.0
            
            # Normalize to -2 to +2 range
            return max(-2.0, min(2.0, score * 1.33))
            
        except (TypeError, ValueError):
            return 0.0
    
    def _extract_sentiment_score(self, finnhub_data: Any) -> float:
        """Extract normalized score from news sentiment."""
        try:
            if finnhub_data is None:
                return 0.0
            
            # Sentiment score from TickerData (-1 to +1)
            if hasattr(finnhub_data, 'sentiment_score'):
                sent = finnhub_data.sentiment_score
                # Scale to -2 to +2 range
                return float(sent) * 2.0
            
            return 0.0
        except (TypeError, ValueError):
            return 0.0
    
    def _extract_momentum_score(
        self,
        price_data: Optional[Dict],
        technical_analysis: Optional[Dict],
    ) -> float:
        """Extract momentum score from price data."""
        try:
            if price_data and 'change_percent' in price_data:
                change = price_data['change_percent']
                # Scale: +5% = +2, -5% = -2
                return max(-2.0, min(2.0, change / 2.5))
            
            if technical_analysis:
                raw = technical_analysis.get('raw_indicators', {})
                # Use ROC (Rate of Change) if available
                volume_ratio = raw.get('Volume_Ratio', 1.0)
                # High volume with direction
                price_info = technical_analysis.get('price', {})
                change_pct = price_info.get('change_percent', 0)
                
                # Volume boost
                boost = 1.0 if volume_ratio < 1.5 else 1.3
                return max(-2.0, min(2.0, (change_pct / 2.5) * boost))
            
            return 0.0
        except (TypeError, ValueError):
            return 0.0
    
    def _classify_signal(self, score: float) -> SignalStrength:
        """Classify score into signal strength."""
        if score >= THRESHOLDS['strong_buy']:
            return SignalStrength.STRONG_BUY
        elif score >= THRESHOLDS['buy']:
            return SignalStrength.BUY
        elif score <= THRESHOLDS['strong_sell']:
            return SignalStrength.STRONG_SELL
        elif score <= THRESHOLDS['sell']:
            return SignalStrength.SELL
        else:
            return SignalStrength.HOLD
    
    def _calculate_confidence(self, components: Dict[str, float]) -> float:
        """Calculate confidence based on component agreement."""
        if not components:
            return 0.0
        
        # Remove metadata keys
        values = [v for k, v in components.items() if not k.startswith('_')]
        
        if len(values) < 2:
            return 0.3  # Low confidence with single signal
        
        # Check if all signals agree in direction
        positive = sum(1 for v in values if v > 0.3)
        negative = sum(1 for v in values if v < -0.3)
        neutral = len(values) - positive - negative
        
        # Agreement ratio
        max_agreement = max(positive, negative, neutral)
        agreement_ratio = max_agreement / len(values)
        
        # Strength factor (stronger signals = higher confidence)
        avg_strength = sum(abs(v) for v in values) / len(values)
        strength_factor = min(1.0, avg_strength / 1.5)
        
        # Combined confidence
        confidence = (agreement_ratio * 0.6 + strength_factor * 0.4)
        
        return min(1.0, confidence)
    
    def get_signal_name(self, score: UnifiedScore) -> str:
        """Get human-readable signal name."""
        return score.signal.name.replace('_', ' ').title()
    
    def get_action_recommendation(self, score: UnifiedScore) -> str:
        """Get actionable recommendation from score."""
        if score.signal == SignalStrength.STRONG_BUY:
            return "STRONG BUY - Multiple indicators bullish"
        elif score.signal == SignalStrength.BUY:
            return "BUY - Majority of signals positive"
        elif score.signal == SignalStrength.STRONG_SELL:
            return "STRONG SELL - Multiple indicators bearish"
        elif score.signal == SignalStrength.SELL:
            return "SELL - Majority of signals negative"
        else:
            return "HOLD - Mixed signals, wait for clarity"


# Convenience function for quick scoring
def calculate_unified_score(
    technical_analysis: Optional[Dict] = None,
    ml_result: Optional[Any] = None,
    finnhub_data: Optional[Any] = None,
    price_data: Optional[Dict] = None,
    include_finnhub: bool = True,
) -> UnifiedScore:
    """
    Convenience function to calculate unified score.
    
    Use this for simple cases. For repeated scoring, create a UnifiedScorer instance.
    """
    scorer = UnifiedScorer(include_finnhub=include_finnhub)
    return scorer.calculate_score(
        technical_analysis=technical_analysis,
        ml_result=ml_result,
        finnhub_data=finnhub_data,
        price_data=price_data,
    )


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)
    
    scorer = UnifiedScorer()
    
    # Test with mock data
    mock_technical = {
        'overall': {
            'signal': 'BUY',
            'signal_value': 1.2,
        },
        'price': {
            'current': 150.0,
            'change_percent': 2.5,
        },
        'raw_indicators': {
            'Volume_Ratio': 1.8,
        }
    }
    
    result = scorer.calculate_score(technical_analysis=mock_technical)
    
    print(f"Score: {result.score}")
    print(f"Signal: {result.signal.name}")
    print(f"Confidence: {result.confidence}")
    print(f"Components: {result.components}")
    print(f"Reason: {result.reason}")
    print(f"Recommendation: {scorer.get_action_recommendation(result)}")
