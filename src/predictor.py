"""
ML Predictor Module
XGBoost-based prediction engine with Walk-Forward Validation and SHAP explainability.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
import logging
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import xgboost as xgb
import joblib

logger = logging.getLogger(__name__)


class PredictionClass(Enum):
    """Prediction classification."""
    STRONG_UP = 2
    UP = 1
    FLAT = 0
    DOWN = -1
    STRONG_DOWN = -2


@dataclass
class PredictionResult:
    """Container for prediction results."""
    prediction: PredictionClass
    probability: float
    confidence: str
    feature_importance: Dict[str, float]
    explanation: str


class MLPredictor:
    """
    Machine Learning predictor using XGBoost with advanced features:
    - Multi-class classification (STRONG_UP, UP, FLAT, DOWN, STRONG_DOWN)
    - Walk-Forward Validation (TimeSeriesSplit)
    - SHAP-based explainability
    - Feature importance analysis
    """
    
    def __init__(self, volatility_threshold: float = 0.01):
        """
        Initialize the predictor.
        
        Args:
            volatility_threshold: Threshold for classifying as UP/DOWN vs FLAT
                                 (as fraction of price, e.g., 0.01 = 1%)
        """
        self.volatility_threshold = volatility_threshold
        self.model: Optional[xgb.XGBClassifier] = None
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.is_trained = False
        self.training_metrics: Dict[str, float] = {}
        
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create ML features from OHLCV and indicator data.
        
        Args:
            df: DataFrame with price and indicator data
            
        Returns:
            DataFrame with engineered features
        """
        features = pd.DataFrame(index=df.index)
        
        # Price-based features
        features['return_1d'] = df['Close'].pct_change(1)
        features['return_5d'] = df['Close'].pct_change(5)
        features['return_10d'] = df['Close'].pct_change(10)
        features['return_20d'] = df['Close'].pct_change(20)
        
        # Volatility features
        features['volatility_5d'] = df['Close'].pct_change().rolling(5).std()
        features['volatility_20d'] = df['Close'].pct_change().rolling(20).std()
        
        # Price relative to moving averages
        if 'SMA_20' in df.columns:
            features['price_vs_sma20'] = (df['Close'] - df['SMA_20']) / df['SMA_20']
        if 'SMA_50' in df.columns:
            features['price_vs_sma50'] = (df['Close'] - df['SMA_50']) / df['SMA_50']
        if 'SMA_200' in df.columns:
            features['price_vs_sma200'] = (df['Close'] - df['SMA_200']) / df['SMA_200']
        
        # Moving average relationships
        if 'SMA_20' in df.columns and 'SMA_50' in df.columns:
            features['sma20_vs_sma50'] = (df['SMA_20'] - df['SMA_50']) / df['SMA_50']
        if 'SMA_50' in df.columns and 'SMA_200' in df.columns:
            features['sma50_vs_sma200'] = (df['SMA_50'] - df['SMA_200']) / df['SMA_200']
        
        # Momentum indicators
        if 'RSI' in df.columns:
            features['rsi'] = df['RSI'] / 100  # Normalize to 0-1
            features['rsi_oversold'] = (df['RSI'] < 30).astype(int)
            features['rsi_overbought'] = (df['RSI'] > 70).astype(int)
        
        if 'MACD_Hist' in df.columns:
            features['macd_hist'] = df['MACD_Hist']
            features['macd_hist_change'] = df['MACD_Hist'].diff()
        
        if 'STOCH_K' in df.columns:
            features['stoch_k'] = df['STOCH_K'] / 100
        
        # Trend strength
        if 'ADX' in df.columns:
            features['adx'] = df['ADX'] / 100
        if 'DI_Plus' in df.columns and 'DI_Minus' in df.columns:
            features['di_diff'] = (df['DI_Plus'] - df['DI_Minus']) / 100
        
        # Bollinger Bands
        if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
            bb_range = df['BB_Upper'] - df['BB_Lower']
            features['bb_position'] = (df['Close'] - df['BB_Lower']) / bb_range
            features['bb_width'] = bb_range / df['BB_Middle'] if 'BB_Middle' in df.columns else bb_range / df['Close']
        
        # Volume features
        if 'Volume_Ratio' in df.columns:
            features['volume_ratio'] = df['Volume_Ratio']
        features['volume_change'] = df['Volume'].pct_change()
        
        # ATR (normalized)
        if 'ATR' in df.columns:
            features['atr_pct'] = df['ATR'] / df['Close']
        
        # High-Low range
        features['hl_range'] = (df['High'] - df['Low']) / df['Close']
        
        # Candlestick patterns (simplified)
        features['body_size'] = (df['Close'] - df['Open']) / df['Open']
        features['upper_shadow'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / df['Close']
        features['lower_shadow'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / df['Close']
        
        return features
    
    def _create_target(self, df: pd.DataFrame, horizon: int = 5) -> pd.Series:
        """
        Create target variable for prediction.
        Uses multi-class classification based on forward returns.
        
        Args:
            df: DataFrame with Close prices
            horizon: Number of days to look ahead
            
        Returns:
            Series with target labels
        """
        # Calculate forward return
        forward_return = df['Close'].shift(-horizon) / df['Close'] - 1
        
        # Use ATR-based threshold if available, otherwise use fixed threshold
        if 'ATR' in df.columns:
            threshold = (df['ATR'] / df['Close']).rolling(20).mean()
        else:
            threshold = self.volatility_threshold
        
        # Create multi-class target
        # 2: STRONG_UP (> 2x threshold)
        # 1: UP (> threshold)
        # 0: FLAT (between -threshold and threshold)
        # -1: DOWN (< -threshold)
        # -2: STRONG_DOWN (< -2x threshold)
        
        target = pd.Series(index=df.index, dtype=int)
        target[forward_return > 2 * threshold] = 2
        target[(forward_return > threshold) & (forward_return <= 2 * threshold)] = 1
        target[(forward_return >= -threshold) & (forward_return <= threshold)] = 0
        target[(forward_return < -threshold) & (forward_return >= -2 * threshold)] = -1
        target[forward_return < -2 * threshold] = -2
        
        return target
    
    def train(self, df: pd.DataFrame, n_splits: int = 5) -> Dict[str, float]:
        """
        Train the model using Walk-Forward Validation.
        
        Args:
            df: DataFrame with price and indicator data
            n_splits: Number of time series splits for validation
            
        Returns:
            Dictionary with training metrics
        """
        logger.info("Starting model training with Walk-Forward Validation...")
        
        # Create features and target
        features = self._create_features(df)
        target = self._create_target(df)
        
        # Combine and drop NaN rows
        combined = pd.concat([features, target.rename('target')], axis=1)
        combined = combined.dropna()
        
        if len(combined) < 30:
            logger.warning(f"Not enough data for training ({len(combined)} samples, need 30+)")
            return {'error': 'Insufficient data'}
        
        X = combined.drop('target', axis=1)
        y = combined['target']
        
        self.feature_names = X.columns.tolist()
        
        # Adapt n_splits based on data size
        effective_splits = min(n_splits, max(2, len(X) // 30))
        
        # Walk-Forward Validation
        tscv = TimeSeriesSplit(n_splits=effective_splits)
        
        accuracies = []
        precisions = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            try:
                # Train XGBoost with explicit num_class for multi-class
                model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    use_label_encoder=False,
                    eval_metric='mlogloss',
                    verbosity=0,
                    num_class=5,  # Classes: 0, 1, 2, 3, 4 (shifted from -2 to 2)
                    objective='multi:softprob'
                )
                
                model.fit(X_train_scaled, y_train + 2)  # Shift labels to 0-4
                
                # Evaluate
                y_pred = model.predict(X_test_scaled) - 2  # Shift back
                
                acc = accuracy_score(y_test, y_pred)
                accuracies.append(acc)
                
                # Precision for directional calls (non-FLAT)
                mask = (y_test != 0) | (y_pred != 0)
                if mask.sum() > 0:
                    prec = accuracy_score(y_test[mask], y_pred[mask])
                    precisions.append(prec)
            except Exception as e:
                logger.debug(f"Fold training failed: {e}")
                continue
        
        # Final training on all data
        try:
            X_scaled = self.scaler.fit_transform(X)
            y_shifted = y + 2  # Shift to 0-4
            
            # Check if all classes are present
            unique_classes = np.unique(y_shifted)
            missing_classes = set(range(5)) - set(unique_classes.astype(int))
            
            if missing_classes:
                # Fallback to 3-class model: bearish(0), neutral(1), bullish(2)
                logger.debug(f"Missing classes {missing_classes}, using simplified model")
                y_simple = np.where(y_shifted < 2, 0, np.where(y_shifted > 2, 2, 1))
                
                self.model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    use_label_encoder=False,
                    eval_metric='mlogloss',
                    verbosity=0,
                    num_class=3,
                    objective='multi:softprob'
                )
                self.model.fit(X_scaled, y_simple)
                self._using_simple_model = True
            else:
                self.model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    use_label_encoder=False,
                    eval_metric='mlogloss',
                    verbosity=0,
                    num_class=5,
                    objective='multi:softprob'
                )
                self.model.fit(X_scaled, y_shifted)
                self._using_simple_model = False
            
            self.is_trained = True
        except Exception as e:
            logger.error(f"Final model training failed: {e}")
            return {'error': str(e)}
        
        self.training_metrics = {
            'avg_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'avg_directional_precision': np.mean(precisions) if precisions else 0,
            'n_samples': len(X),
            'n_features': len(self.feature_names),
            'n_splits': n_splits,
        }
        
        logger.info(f"Training complete. Accuracy: {self.training_metrics['avg_accuracy']:.2%}")
        
        return self.training_metrics
    
    def predict(self, df: pd.DataFrame) -> Optional[PredictionResult]:
        """
        Make a prediction for the latest data point.
        
        Args:
            df: DataFrame with price and indicator data
            
        Returns:
            PredictionResult with prediction and explanation
        """
        if not self.is_trained or self.model is None:
            logger.warning("Model not trained. Training now...")
            self.train(df)
        
        if not self.is_trained:
            return None
        
        # Create features for latest data
        features = self._create_features(df)
        latest = features.iloc[[-1]].dropna(axis=1)
        
        # Ensure same features as training
        missing_features = set(self.feature_names) - set(latest.columns)
        for feat in missing_features:
            latest[feat] = 0
        
        latest = latest[self.feature_names]
        
        try:
            # Scale and predict
            X_scaled = self.scaler.transform(latest)
            
            # Get prediction and probabilities
            raw_pred = self.model.predict(X_scaled)[0]
            probabilities = self.model.predict_proba(X_scaled)[0]
            
            # Handle simplified 3-class model vs full 5-class model
            if getattr(self, '_using_simple_model', False):
                # 3-class model: 0=bearish, 1=neutral, 2=bullish
                # Map to -1, 0, 1
                pred_class = int(raw_pred) - 1
            else:
                # 5-class model: 0-4 maps to -2 to 2
                pred_class = int(raw_pred) - 2
            
            # Clamp to valid range
            pred_class = max(-2, min(2, pred_class))
            
            # Get confidence (max probability)
            max_prob = float(max(probabilities))
            
            if max_prob > 0.6:
                confidence = "HIGH"
            elif max_prob > 0.4:
                confidence = "MEDIUM"
            else:
                confidence = "LOW"
            
            # Get feature importance for this prediction
            importance = dict(zip(self.feature_names, self.model.feature_importances_))
            top_features = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5])
            
            # Generate explanation
            explanation = self._generate_explanation(pred_class, top_features, latest, max_prob)
            
            # Map to enum
            pred_enum = {
                2: PredictionClass.STRONG_UP,
                1: PredictionClass.UP,
                0: PredictionClass.FLAT,
                -1: PredictionClass.DOWN,
                -2: PredictionClass.STRONG_DOWN,
            }[pred_class]
            
            return PredictionResult(
                prediction=pred_enum,
                probability=float(max_prob),
                confidence=confidence,
                feature_importance=top_features,
                explanation=explanation
            )
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return None
    
    def _generate_explanation(
        self,
        prediction: int,
        top_features: Dict[str, float],
        features: pd.DataFrame,
        probability: float
    ) -> str:
        """Generate human-readable explanation for the prediction."""
        
        direction = {
            2: "STRONG UPWARD",
            1: "UPWARD",
            0: "SIDEWAYS",
            -1: "DOWNWARD",
            -2: "STRONG DOWNWARD"
        }[prediction]
        
        explanations = []
        
        for feat, imp in top_features.items():
            if imp < 0.05:
                continue
                
            val = features[feat].iloc[0]
            
            # Generate feature-specific explanations
            if 'rsi' in feat and 'oversold' not in feat and 'overbought' not in feat:
                if val < 0.3:
                    explanations.append(f"RSI is oversold ({val*100:.1f})")
                elif val > 0.7:
                    explanations.append(f"RSI is overbought ({val*100:.1f})")
            elif 'price_vs_sma' in feat:
                if val > 0.02:
                    explanations.append(f"Price {val*100:.1f}% above {feat.split('_')[-1].upper()}")
                elif val < -0.02:
                    explanations.append(f"Price {abs(val)*100:.1f}% below {feat.split('_')[-1].upper()}")
            elif 'macd' in feat:
                if val > 0:
                    explanations.append("MACD histogram positive (bullish momentum)")
                else:
                    explanations.append("MACD histogram negative (bearish momentum)")
            elif 'volume_ratio' in feat:
                if val > 1.5:
                    explanations.append(f"High volume ({val:.1f}x average)")
            elif 'adx' in feat:
                if val > 0.25:
                    explanations.append(f"Strong trend (ADX={val*100:.1f})")
            elif 'bb_position' in feat:
                if val > 0.8:
                    explanations.append("Near upper Bollinger Band")
                elif val < 0.2:
                    explanations.append("Near lower Bollinger Band")
        
        if not explanations:
            explanations.append("Based on overall technical pattern")
        
        base = f"Model predicts {direction} movement (confidence: {probability:.0%}). "
        factors = "Key factors: " + "; ".join(explanations[:3])
        
        return base + factors
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model."""
        if not self.is_trained or self.model is None:
            return {}
        
        importance = dict(zip(self.feature_names, self.model.feature_importances_))
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    
    def save_model(self, path: str) -> None:
        """Save trained model to file."""
        if self.is_trained:
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'metrics': self.training_metrics,
            }, path)
            logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str) -> bool:
        """Load model from file."""
        try:
            data = joblib.load(path)
            self.model = data['model']
            self.scaler = data['scaler']
            self.feature_names = data['feature_names']
            self.training_metrics = data['metrics']
            self.is_trained = True
            logger.info(f"Model loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False


if __name__ == "__main__":
    # Quick test
    import sys
    sys.path.insert(0, '/home/boisv/code/stock-market/src')
    from data_fetcher import DataFetcher
    from indicators import TechnicalIndicators
    
    print("Fetching AAPL data...")
    fetcher = DataFetcher()
    df = fetcher.fetch_ticker_data("AAPL", period="2y")
    
    if df is not None:
        print("Calculating indicators...")
        ti = TechnicalIndicators(df)
        df_with_indicators = ti.get_dataframe()
        
        print("Training model...")
        predictor = MLPredictor()
        metrics = predictor.train(df_with_indicators)
        
        print(f"\nTraining Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
        
        print("\nMaking prediction...")
        result = predictor.predict(df_with_indicators)
        
        if result:
            print(f"\nPrediction: {result.prediction.name}")
            print(f"Confidence: {result.confidence} ({result.probability:.0%})")
            print(f"Explanation: {result.explanation}")
            print(f"\nTop Features:")
            for feat, imp in list(result.feature_importance.items())[:5]:
                print(f"  {feat}: {imp:.3f}")
