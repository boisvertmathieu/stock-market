"""
Technical Indicators Module
Calculates technical analysis indicators using pandas-ta.
"""

import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class Signal(Enum):
    """Trading signal strength."""
    STRONG_BUY = 2
    BUY = 1
    HOLD = 0
    SELL = -1
    STRONG_SELL = -2


@dataclass
class IndicatorResult:
    """Container for indicator calculation results."""
    name: str
    value: float
    signal: Signal
    description: str


class TechnicalIndicators:
    """
    Calculates and analyzes technical indicators for stock data.
    Uses pandas-ta for efficient indicator calculations.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with OHLCV DataFrame.
        
        Args:
            df: DataFrame with Open, High, Low, Close, Volume columns
        """
        self.df = df.copy()
        self._calculate_all_indicators()
    
    def _calculate_all_indicators(self) -> None:
        """Calculate all technical indicators at once."""
        df = self.df
        
        # Moving Averages
        df['SMA_20'] = ta.sma(df['Close'], length=20)
        df['SMA_50'] = ta.sma(df['Close'], length=50)
        df['SMA_200'] = ta.sma(df['Close'], length=200)
        df['EMA_12'] = ta.ema(df['Close'], length=12)
        df['EMA_26'] = ta.ema(df['Close'], length=26)
        
        # RSI
        df['RSI'] = ta.rsi(df['Close'], length=14)
        
        # MACD
        macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
        if macd is not None:
            df['MACD'] = macd['MACD_12_26_9']
            df['MACD_Signal'] = macd['MACDs_12_26_9']
            df['MACD_Hist'] = macd['MACDh_12_26_9']
        
        # Bollinger Bands
        bbands = ta.bbands(df['Close'], length=20, std=2)
        if bbands is not None and len(bbands.columns) >= 3:
            # pandas-ta uses different naming conventions depending on version
            bb_cols = bbands.columns.tolist()
            # Find upper, middle, lower bands by prefix
            upper_col = next((c for c in bb_cols if c.startswith('BBU')), None)
            middle_col = next((c for c in bb_cols if c.startswith('BBM')), None)
            lower_col = next((c for c in bb_cols if c.startswith('BBL')), None)
            
            if upper_col and middle_col and lower_col:
                df['BB_Upper'] = bbands[upper_col]
                df['BB_Middle'] = bbands[middle_col]
                df['BB_Lower'] = bbands[lower_col]
                df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']

        
        # ATR (Volatility)
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        
        # Stochastic
        stoch = ta.stoch(df['High'], df['Low'], df['Close'])
        if stoch is not None:
            df['STOCH_K'] = stoch['STOCHk_14_3_3']
            df['STOCH_D'] = stoch['STOCHd_14_3_3']
        
        # Volume indicators
        df['Volume_SMA_20'] = ta.sma(df['Volume'], length=20)
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_20']
        
        # ADX (Trend Strength)
        adx = ta.adx(df['High'], df['Low'], df['Close'], length=14)
        if adx is not None:
            df['ADX'] = adx['ADX_14']
            df['DI_Plus'] = adx['DMP_14']
            df['DI_Minus'] = adx['DMN_14']
        
        # OBV (On-Balance Volume)
        df['OBV'] = ta.obv(df['Close'], df['Volume'])
        
        self.df = df
    
    def get_latest_indicators(self) -> Dict[str, float]:
        """Get the most recent values for all indicators."""
        latest = self.df.iloc[-1]
        
        indicators = {}
        for col in self.df.columns:
            if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']:
                val = latest[col]
                if pd.notna(val):
                    indicators[col] = round(float(val), 4)
                    
        return indicators
    
    def analyze_rsi(self) -> IndicatorResult:
        """Analyze RSI indicator."""
        rsi = self.df['RSI'].iloc[-1]
        
        if pd.isna(rsi):
            return IndicatorResult("RSI", 50, Signal.HOLD, "Insufficient data")
        
        if rsi < 30:
            signal = Signal.STRONG_BUY
            desc = f"RSI {rsi:.1f} - Oversold (< 30)"
        elif rsi < 40:
            signal = Signal.BUY
            desc = f"RSI {rsi:.1f} - Approaching oversold"
        elif rsi > 70:
            signal = Signal.STRONG_SELL
            desc = f"RSI {rsi:.1f} - Overbought (> 70)"
        elif rsi > 60:
            signal = Signal.SELL
            desc = f"RSI {rsi:.1f} - Approaching overbought"
        else:
            signal = Signal.HOLD
            desc = f"RSI {rsi:.1f} - Neutral zone"
            
        return IndicatorResult("RSI", rsi, signal, desc)
    
    def analyze_macd(self) -> IndicatorResult:
        """Analyze MACD indicator."""
        if 'MACD' not in self.df.columns:
            return IndicatorResult("MACD", 0, Signal.HOLD, "MACD not available")
        
        macd = self.df['MACD'].iloc[-1]
        signal_line = self.df['MACD_Signal'].iloc[-1]
        hist = self.df['MACD_Hist'].iloc[-1]
        prev_hist = self.df['MACD_Hist'].iloc[-2] if len(self.df) > 1 else hist
        
        if pd.isna(macd) or pd.isna(signal_line):
            return IndicatorResult("MACD", 0, Signal.HOLD, "Insufficient data")
        
        # Check for crossovers
        if hist > 0 and prev_hist <= 0:
            signal = Signal.STRONG_BUY
            desc = "MACD Bullish Crossover (Golden Cross)"
        elif hist < 0 and prev_hist >= 0:
            signal = Signal.STRONG_SELL
            desc = "MACD Bearish Crossover (Death Cross)"
        elif hist > 0:
            signal = Signal.BUY
            desc = f"MACD Bullish ({hist:.2f})"
        elif hist < 0:
            signal = Signal.SELL
            desc = f"MACD Bearish ({hist:.2f})"
        else:
            signal = Signal.HOLD
            desc = "MACD Neutral"
            
        return IndicatorResult("MACD", hist, signal, desc)
    
    def analyze_moving_averages(self) -> IndicatorResult:
        """Analyze moving average crossovers and position."""
        close = self.df['Close'].iloc[-1]
        sma20 = self.df['SMA_20'].iloc[-1]
        sma50 = self.df['SMA_50'].iloc[-1]
        sma200 = self.df['SMA_200'].iloc[-1]
        
        if pd.isna(sma200):
            # Not enough data for SMA200
            if pd.isna(sma50):
                return IndicatorResult("MA", close, Signal.HOLD, "Insufficient data for MA analysis")
            # Use SMA20/SMA50 only
            if sma20 > sma50 and close > sma20:
                return IndicatorResult("MA", close, Signal.BUY, f"Price above rising SMAs")
            elif sma20 < sma50 and close < sma20:
                return IndicatorResult("MA", close, Signal.SELL, f"Price below falling SMAs")
            else:
                return IndicatorResult("MA", close, Signal.HOLD, "Mixed MA signals")
        
        # Full analysis with SMA200
        bullish_count = 0
        bearish_count = 0
        
        # Price vs SMAs
        if close > sma20: bullish_count += 1
        else: bearish_count += 1
        if close > sma50: bullish_count += 1
        else: bearish_count += 1
        if close > sma200: bullish_count += 1
        else: bearish_count += 1
        
        # SMA ordering (ideal bullish: SMA20 > SMA50 > SMA200)
        if sma20 > sma50 > sma200:
            bullish_count += 2
            order = "Perfect Bullish Order"
        elif sma20 < sma50 < sma200:
            bearish_count += 2
            order = "Perfect Bearish Order"
        else:
            order = "Mixed"
        
        # Golden/Death Cross detection
        prev_sma50 = self.df['SMA_50'].iloc[-2] if len(self.df) > 1 else sma50
        prev_sma200 = self.df['SMA_200'].iloc[-2] if len(self.df) > 1 else sma200
        
        if sma50 > sma200 and prev_sma50 <= prev_sma200:
            return IndicatorResult("MA", close, Signal.STRONG_BUY, "Golden Cross! SMA50 crossed above SMA200")
        elif sma50 < sma200 and prev_sma50 >= prev_sma200:
            return IndicatorResult("MA", close, Signal.STRONG_SELL, "Death Cross! SMA50 crossed below SMA200")
        
        # Determine signal from counts
        if bullish_count >= 4:
            signal = Signal.STRONG_BUY
        elif bullish_count >= 3:
            signal = Signal.BUY
        elif bearish_count >= 4:
            signal = Signal.STRONG_SELL
        elif bearish_count >= 3:
            signal = Signal.SELL
        else:
            signal = Signal.HOLD
            
        desc = f"MA Analysis ({order}): Price {'above' if close > sma50 else 'below'} SMA50"
        return IndicatorResult("MA", close, signal, desc)
    
    def analyze_bollinger_bands(self) -> IndicatorResult:
        """Analyze Bollinger Bands position."""
        if 'BB_Upper' not in self.df.columns:
            return IndicatorResult("BB", 0, Signal.HOLD, "Bollinger Bands not available")
        
        close = self.df['Close'].iloc[-1]
        upper = self.df['BB_Upper'].iloc[-1]
        lower = self.df['BB_Lower'].iloc[-1]
        middle = self.df['BB_Middle'].iloc[-1]
        
        if pd.isna(upper):
            return IndicatorResult("BB", 0, Signal.HOLD, "Insufficient data")
        
        # Calculate position within bands (0 = lower, 1 = upper)
        position = (close - lower) / (upper - lower) if (upper - lower) != 0 else 0.5
        
        if close >= upper:
            signal = Signal.SELL
            desc = f"Price at/above Upper Band - Potential overbought"
        elif close <= lower:
            signal = Signal.BUY
            desc = f"Price at/below Lower Band - Potential oversold"
        elif position > 0.8:
            signal = Signal.SELL
            desc = f"Price near Upper Band ({position:.0%}) - Resistance"
        elif position < 0.2:
            signal = Signal.BUY
            desc = f"Price near Lower Band ({position:.0%}) - Support"
        else:
            signal = Signal.HOLD
            desc = f"Price within bands ({position:.0%})"
            
        return IndicatorResult("BB", position, signal, desc)
    
    def analyze_volume(self) -> IndicatorResult:
        """Analyze volume trends."""
        if 'Volume_Ratio' not in self.df.columns:
            return IndicatorResult("Volume", 0, Signal.HOLD, "Volume data not available")
        
        ratio = self.df['Volume_Ratio'].iloc[-1]
        price_change = (self.df['Close'].iloc[-1] - self.df['Close'].iloc[-2]) / self.df['Close'].iloc[-2]
        
        if pd.isna(ratio):
            return IndicatorResult("Volume", 0, Signal.HOLD, "Insufficient data")
        
        # High volume with price direction confirms trend
        if ratio > 2.0:
            if price_change > 0.01:
                signal = Signal.STRONG_BUY
                desc = f"Very High Volume ({ratio:.1f}x) with price up - Strong buying"
            elif price_change < -0.01:
                signal = Signal.STRONG_SELL
                desc = f"Very High Volume ({ratio:.1f}x) with price down - Strong selling"
            else:
                signal = Signal.HOLD
                desc = f"Very High Volume ({ratio:.1f}x) - Watch for breakout"
        elif ratio > 1.5:
            if price_change > 0.005:
                signal = Signal.BUY
                desc = f"Above avg Volume ({ratio:.1f}x) with price up"
            elif price_change < -0.005:
                signal = Signal.SELL
                desc = f"Above avg Volume ({ratio:.1f}x) with price down"
            else:
                signal = Signal.HOLD
                desc = f"Above avg Volume ({ratio:.1f}x)"
        else:
            signal = Signal.HOLD
            desc = f"Normal Volume ({ratio:.1f}x avg)"
            
        return IndicatorResult("Volume", ratio, signal, desc)
    
    def analyze_trend_strength(self) -> IndicatorResult:
        """Analyze ADX for trend strength."""
        if 'ADX' not in self.df.columns:
            return IndicatorResult("ADX", 0, Signal.HOLD, "ADX not available")
        
        adx = self.df['ADX'].iloc[-1]
        di_plus = self.df['DI_Plus'].iloc[-1]
        di_minus = self.df['DI_Minus'].iloc[-1]
        
        if pd.isna(adx):
            return IndicatorResult("ADX", 0, Signal.HOLD, "Insufficient data")
        
        # ADX > 25 indicates strong trend
        if adx > 25:
            if di_plus > di_minus:
                signal = Signal.BUY if adx < 40 else Signal.STRONG_BUY
                desc = f"Strong Uptrend (ADX={adx:.1f}, +DI > -DI)"
            else:
                signal = Signal.SELL if adx < 40 else Signal.STRONG_SELL
                desc = f"Strong Downtrend (ADX={adx:.1f}, -DI > +DI)"
        else:
            signal = Signal.HOLD
            desc = f"Weak/No Trend (ADX={adx:.1f})"
            
        return IndicatorResult("ADX", adx, signal, desc)
    
    def analyze_stochastic(self) -> IndicatorResult:
        """Analyze Stochastic oscillator."""
        if 'STOCH_K' not in self.df.columns:
            return IndicatorResult("Stoch", 0, Signal.HOLD, "Stochastic not available")
        
        k = self.df['STOCH_K'].iloc[-1]
        d = self.df['STOCH_D'].iloc[-1]
        
        if pd.isna(k):
            return IndicatorResult("Stoch", 0, Signal.HOLD, "Insufficient data")
        
        if k < 20 and d < 20:
            if k > d:  # K crossing above D
                signal = Signal.STRONG_BUY
                desc = f"Stoch Oversold + Bullish Cross (K={k:.1f})"
            else:
                signal = Signal.BUY
                desc = f"Stoch Oversold (K={k:.1f})"
        elif k > 80 and d > 80:
            if k < d:  # K crossing below D
                signal = Signal.STRONG_SELL
                desc = f"Stoch Overbought + Bearish Cross (K={k:.1f})"
            else:
                signal = Signal.SELL
                desc = f"Stoch Overbought (K={k:.1f})"
        else:
            signal = Signal.HOLD
            desc = f"Stoch Neutral (K={k:.1f})"
            
        return IndicatorResult("Stoch", k, signal, desc)
    
    def get_comprehensive_analysis(self) -> Dict[str, Any]:
        """
        Get comprehensive technical analysis with all indicators.
        Returns a dictionary with all analyses and overall recommendation.
        """
        analyses = {
            'rsi': self.analyze_rsi(),
            'macd': self.analyze_macd(),
            'moving_averages': self.analyze_moving_averages(),
            'bollinger_bands': self.analyze_bollinger_bands(),
            'volume': self.analyze_volume(),
            'trend_strength': self.analyze_trend_strength(),
            'stochastic': self.analyze_stochastic(),
        }
        
        # Calculate overall score
        total_score = sum(a.signal.value for a in analyses.values())
        num_indicators = len(analyses)
        avg_score = total_score / num_indicators
        
        # Determine overall recommendation
        if avg_score >= 1.5:
            overall = Signal.STRONG_BUY
            action = "STRONG BUY - Multiple indicators align bullish"
        elif avg_score >= 0.5:
            overall = Signal.BUY
            action = "BUY - Majority of indicators bullish"
        elif avg_score <= -1.5:
            overall = Signal.STRONG_SELL
            action = "STRONG SELL - Multiple indicators align bearish"
        elif avg_score <= -0.5:
            overall = Signal.SELL
            action = "SELL - Majority of indicators bearish"
        else:
            overall = Signal.HOLD
            action = "HOLD - Mixed signals, wait for clearer direction"
        
        # Get current price info
        close = self.df['Close'].iloc[-1]
        prev_close = self.df['Close'].iloc[-2] if len(self.df) > 1 else close
        change = close - prev_close
        change_pct = (change / prev_close) * 100 if prev_close != 0 else 0
        
        return {
            'price': {
                'current': round(close, 2),
                'change': round(change, 2),
                'change_percent': round(change_pct, 2),
            },
            'indicators': {name: {
                'value': round(a.value, 2) if isinstance(a.value, float) else a.value,
                'signal': a.signal.name,
                'signal_value': a.signal.value,
                'description': a.description
            } for name, a in analyses.items()},
            'overall': {
                'signal': overall.name,
                'signal_value': overall.value,
                'score': round(avg_score, 2),
                'action': action,
            },
            'raw_indicators': self.get_latest_indicators()
        }
    
    def get_dataframe(self) -> pd.DataFrame:
        """Return the DataFrame with all calculated indicators."""
        return self.df


if __name__ == "__main__":
    # Quick test
    import yfinance as yf
    
    print("Fetching AAPL data...")
    stock = yf.Ticker("AAPL")
    df = stock.history(period="1y")
    
    print("Calculating indicators...")
    ti = TechnicalIndicators(df)
    
    analysis = ti.get_comprehensive_analysis()
    
    print(f"\nAAPL Analysis:")
    print(f"Price: ${analysis['price']['current']} ({analysis['price']['change_percent']:+.2f}%)")
    print(f"\nOverall: {analysis['overall']['action']}")
    print(f"\nIndicators:")
    for name, data in analysis['indicators'].items():
        print(f"  {name}: {data['signal']} - {data['description']}")
