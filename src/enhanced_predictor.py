"""
Enhanced ML Predictor Module
With Optuna hyperparameter optimization, SHAP explainability, and fundamental data.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb
import joblib

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

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
    shap_values: Optional[Dict[str, float]] = None


class EnhancedMLPredictor:
    """
    Enhanced ML predictor with:
    - Optuna hyperparameter optimization
    - SHAP explainability
    - Fundamental data integration
    - Multi-class classification with volatility-adjusted thresholds
    """
    
    def __init__(
        self,
        volatility_threshold: float = 0.01,
        optimize_hyperparams: bool = True,
        use_shap: bool = True,
    ):
        """
        Initialize the enhanced predictor.
        
        Args:
            volatility_threshold: Base threshold for UP/DOWN classification
            optimize_hyperparams: Whether to use Optuna optimization
            use_shap: Whether to compute SHAP values
        """
        self.volatility_threshold = volatility_threshold
        self.optimize_hyperparams = optimize_hyperparams and OPTUNA_AVAILABLE
        self.use_shap = use_shap and SHAP_AVAILABLE
        
        self.model: Optional[xgb.XGBClassifier] = None
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.is_trained = False
        self.training_metrics: Dict[str, float] = {}
        self.best_params: Dict[str, Any] = {}
        self.shap_explainer = None
        
    def _create_features(self, df: pd.DataFrame, fundamental_data: Optional[Dict] = None) -> pd.DataFrame:
        """
        Create ML features including technical and fundamental indicators.
        """
        features = pd.DataFrame(index=df.index)
        
        # === Price-based features ===
        features['return_1d'] = df['Close'].pct_change(1)
        features['return_3d'] = df['Close'].pct_change(3)
        features['return_5d'] = df['Close'].pct_change(5)
        features['return_10d'] = df['Close'].pct_change(10)
        features['return_20d'] = df['Close'].pct_change(20)
        
        # === Volatility features ===
        features['volatility_5d'] = df['Close'].pct_change().rolling(5).std()
        features['volatility_10d'] = df['Close'].pct_change().rolling(10).std()
        features['volatility_20d'] = df['Close'].pct_change().rolling(20).std()
        
        # Volatility ratio (short-term vs long-term)
        features['vol_ratio'] = features['volatility_5d'] / features['volatility_20d'].replace(0, np.nan)
        
        # === Momentum features ===
        # Rate of Change (ROC)
        features['roc_5'] = (df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5)
        features['roc_10'] = (df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)
        features['roc_20'] = (df['Close'] - df['Close'].shift(20)) / df['Close'].shift(20)
        
        # Momentum (price difference)
        features['momentum_10'] = df['Close'] - df['Close'].shift(10)
        features['momentum_20'] = df['Close'] - df['Close'].shift(20)
        
        # Williams %R
        highest_high = df['High'].rolling(14).max()
        lowest_low = df['Low'].rolling(14).min()
        features['williams_r'] = (highest_high - df['Close']) / (highest_high - lowest_low).replace(0, np.nan) * -100
        
        # === Moving Average features ===
        if 'SMA_20' in df.columns:
            features['price_vs_sma20'] = (df['Close'] - df['SMA_20']) / df['SMA_20']
        else:
            features['price_vs_sma20'] = (df['Close'] - df['Close'].rolling(20).mean()) / df['Close'].rolling(20).mean()
            
        if 'SMA_50' in df.columns:
            features['price_vs_sma50'] = (df['Close'] - df['SMA_50']) / df['SMA_50']
        else:
            features['price_vs_sma50'] = (df['Close'] - df['Close'].rolling(50).mean()) / df['Close'].rolling(50).mean()
            
        if 'SMA_200' in df.columns:
            features['price_vs_sma200'] = (df['Close'] - df['SMA_200']) / df['SMA_200']
        else:
            sma200 = df['Close'].rolling(200).mean()
            features['price_vs_sma200'] = (df['Close'] - sma200) / sma200
        
        # MA crossovers (golden cross / death cross signal)
        sma_20 = df['SMA_20'] if 'SMA_20' in df.columns else df['Close'].rolling(20).mean()
        sma_50 = df['SMA_50'] if 'SMA_50' in df.columns else df['Close'].rolling(50).mean()
        features['ma_crossover'] = (sma_20 - sma_50) / sma_50
        
        # === Technical indicators ===
        if 'RSI' in df.columns:
            features['rsi'] = df['RSI'] / 100
            features['rsi_oversold'] = (df['RSI'] < 30).astype(int)
            features['rsi_overbought'] = (df['RSI'] > 70).astype(int)
            # RSI momentum
            features['rsi_momentum'] = df['RSI'].diff(3) / 100
        
        if 'MACD_Hist' in df.columns:
            features['macd_hist'] = df['MACD_Hist']
            features['macd_hist_change'] = df['MACD_Hist'].diff()
            # MACD cross signal
            features['macd_positive'] = (df['MACD_Hist'] > 0).astype(int)
        
        if 'STOCH_K' in df.columns:
            features['stoch_k'] = df['STOCH_K'] / 100
            features['stoch_d'] = df['STOCH_D'] / 100 if 'STOCH_D' in df.columns else 0
        
        # === Trend strength ===
        if 'ADX' in df.columns:
            features['adx'] = df['ADX'] / 100
            features['strong_trend'] = (df['ADX'] > 25).astype(int)
        if 'DI_Plus' in df.columns and 'DI_Minus' in df.columns:
            features['di_diff'] = (df['DI_Plus'] - df['DI_Minus']) / 100
            features['bullish_di'] = (df['DI_Plus'] > df['DI_Minus']).astype(int)
        
        # === Bollinger Bands ===
        if 'BB_Upper' in df.columns and 'BB_Lower' in df.columns:
            bb_range = df['BB_Upper'] - df['BB_Lower']
            features['bb_position'] = (df['Close'] - df['BB_Lower']) / bb_range.replace(0, np.nan)
            features['bb_width'] = bb_range / df['BB_Middle'] if 'BB_Middle' in df.columns else bb_range / df['Close']
            features['bb_squeeze'] = (features['bb_width'] < features['bb_width'].rolling(20).mean()).astype(int)
        
        # === Volume features ===
        if 'Volume_Ratio' in df.columns:
            features['volume_ratio'] = df['Volume_Ratio']
        else:
            features['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        features['volume_trend'] = df['Volume'].pct_change(5)
        features['high_volume'] = (features['volume_ratio'] > 1.5).astype(int)
        
        # === ATR (normalized volatility) ===
        if 'ATR' in df.columns:
            features['atr_pct'] = df['ATR'] / df['Close']
        else:
            tr = pd.concat([
                df['High'] - df['Low'],
                abs(df['High'] - df['Close'].shift(1)),
                abs(df['Low'] - df['Close'].shift(1))
            ], axis=1).max(axis=1)
            features['atr_pct'] = tr.rolling(14).mean() / df['Close']
        
        # === Price patterns ===
        features['hl_range'] = (df['High'] - df['Low']) / df['Close']
        features['body_size'] = abs(df['Close'] - df['Open']) / df['Open']
        features['upper_shadow'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / df['Close']
        features['lower_shadow'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / df['Close']
        features['bullish_candle'] = (df['Close'] > df['Open']).astype(int)
        
        # === Streak features ===
        returns = df['Close'].pct_change()
        pos_streak = (returns > 0).astype(int)
        features['win_streak'] = pos_streak.groupby((pos_streak != pos_streak.shift()).cumsum()).cumsum()
        neg_streak = (returns < 0).astype(int)
        features['loss_streak'] = neg_streak.groupby((neg_streak != neg_streak.shift()).cumsum()).cumsum()
        
        # === Fundamental features (if available) ===
        if fundamental_data:
            pe_ratio = fundamental_data.get('pe_ratio')
            if pe_ratio and not np.isnan(pe_ratio):
                # Normalize PE relative to market average (~20)
                features['pe_zscore'] = (pe_ratio - 20) / 10
            
            peg_ratio = fundamental_data.get('peg_ratio')
            if peg_ratio and not np.isnan(peg_ratio):
                features['peg_ratio'] = min(peg_ratio, 5) / 5  # Cap at 5
            
            revenue_growth = fundamental_data.get('revenue_growth')
            if revenue_growth and not np.isnan(revenue_growth):
                features['revenue_growth'] = revenue_growth
            
            profit_margin = fundamental_data.get('profit_margin')
            if profit_margin and not np.isnan(profit_margin):
                features['profit_margin'] = profit_margin
        
        return features
    
    def _create_target(self, df: pd.DataFrame, horizon: int = 5) -> pd.Series:
        """
        Create target with volatility-adjusted thresholds.
        Uses asymmetric thresholds for better signal quality.
        """
        forward_return = df['Close'].shift(-horizon) / df['Close'] - 1
        
        # Use rolling volatility for adaptive thresholds
        volatility = df['Close'].pct_change().rolling(20).std()
        
        # Base threshold scaled by volatility
        threshold = np.maximum(volatility * 1.5, self.volatility_threshold)
        strong_threshold = threshold * 2
        
        target = pd.Series(index=df.index, dtype=float)
        target[forward_return > strong_threshold] = 2  # STRONG_UP
        target[(forward_return > threshold) & (forward_return <= strong_threshold)] = 1  # UP
        target[(forward_return >= -threshold) & (forward_return <= threshold)] = 0  # FLAT
        target[(forward_return < -threshold) & (forward_return >= -strong_threshold)] = -1  # DOWN
        target[forward_return < -strong_threshold] = -2  # STRONG_DOWN
        
        return target
    
    def _optimize_with_optuna(self, X: np.ndarray, y: np.ndarray, n_trials: int = 30) -> Dict[str, Any]:
        """
        Optimize XGBoost hyperparameters using Optuna.
        """
        if not OPTUNA_AVAILABLE:
            return self._get_default_params()
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 0.5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 2),
            }
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model = xgb.XGBClassifier(
                    **params,
                    random_state=42,
                    use_label_encoder=False,
                    eval_metric='mlogloss',
                    verbosity=0,
                    num_class=5,
                    objective='multi:softprob'
                )
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                
                # Score: weighted accuracy (directional predictions matter more)
                acc = accuracy_score(y_val, y_pred)
                
                # Bonus for directional accuracy
                directional_mask = (y_val != 2) | (y_pred != 2)  # Non-FLAT
                if directional_mask.sum() > 0:
                    dir_acc = accuracy_score(y_val[directional_mask], y_pred[directional_mask])
                    score = 0.5 * acc + 0.5 * dir_acc
                else:
                    score = acc
                
                scores.append(score)
            
            return np.mean(scores)
        
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        return study.best_params
    
    def _get_default_params(self) -> Dict[str, Any]:
        """Get default XGBoost parameters."""
        return {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
        }
    
    def train(
        self,
        df: pd.DataFrame,
        fundamental_data: Optional[Dict] = None,
        n_splits: int = 3,
        n_optuna_trials: int = 30
    ) -> Dict[str, float]:
        """
        Train the model with optional hyperparameter optimization.
        """
        # Create features and target
        features = self._create_features(df, fundamental_data)
        target = self._create_target(df)
        
        # Combine and drop NaN
        combined = pd.concat([features, target.rename('target')], axis=1)
        combined = combined.dropna()
        
        if len(combined) < 30:
            logger.warning(f"Not enough data for training ({len(combined)} samples)")
            return {'error': 'Insufficient data'}
        
        X = combined.drop('target', axis=1)
        y = combined['target']
        
        self.feature_names = X.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        y_shifted = y.values + 2  # Shift to 0-4
        
        # Optimize hyperparameters
        if self.optimize_hyperparams and len(X) >= 50:
            try:
                self.best_params = self._optimize_with_optuna(X_scaled, y_shifted, n_optuna_trials)
            except Exception as e:
                logger.debug(f"Optuna optimization failed: {e}")
                self.best_params = self._get_default_params()
        else:
            self.best_params = self._get_default_params()
        
        # Train final model
        self.model = xgb.XGBClassifier(
            **self.best_params,
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss',
            verbosity=0,
            num_class=5,
            objective='multi:softprob'
        )
        
        try:
            self.model.fit(X_scaled, y_shifted)
            self.is_trained = True
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return {'error': str(e)}
        
        # Initialize SHAP explainer
        if self.use_shap and self.model is not None:
            try:
                self.shap_explainer = shap.TreeExplainer(self.model)
            except:
                self.shap_explainer = None
        
        # Calculate training metrics via cross-validation
        tscv = TimeSeriesSplit(n_splits=n_splits)
        accuracies = []
        
        for train_idx, test_idx in tscv.split(X_scaled):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = y_shifted[train_idx], y_shifted[test_idx]
            
            try:
                temp_model = xgb.XGBClassifier(**self.best_params, random_state=42,
                    use_label_encoder=False, eval_metric='mlogloss', verbosity=0,
                    num_class=5, objective='multi:softprob')
                temp_model.fit(X_train, y_train)
                y_pred = temp_model.predict(X_test)
                accuracies.append(accuracy_score(y_test, y_pred))
            except:
                continue
        
        self.training_metrics = {
            'avg_accuracy': np.mean(accuracies) if accuracies else 0,
            'n_samples': len(X),
            'n_features': len(self.feature_names),
            'optimized': self.optimize_hyperparams,
        }
        
        return self.training_metrics
    
    def predict(self, df: pd.DataFrame, fundamental_data: Optional[Dict] = None) -> Optional[PredictionResult]:
        """
        Make prediction with SHAP explanations.
        """
        if not self.is_trained or self.model is None:
            return None
        
        # Create features
        features = self._create_features(df, fundamental_data)
        latest = features.iloc[[-1]].dropna(axis=1)
        
        # Ensure same features
        missing = set(self.feature_names) - set(latest.columns)
        for feat in missing:
            latest[feat] = 0
        
        extra = set(latest.columns) - set(self.feature_names)
        latest = latest.drop(columns=list(extra), errors='ignore')
        latest = latest[self.feature_names]
        
        try:
            X_scaled = self.scaler.transform(latest)
            
            pred_class = self.model.predict(X_scaled)[0] - 2
            probabilities = self.model.predict_proba(X_scaled)[0]
            max_prob = max(probabilities)
            
            confidence = "HIGH" if max_prob > 0.6 else "MEDIUM" if max_prob > 0.4 else "LOW"
            
            # Feature importance
            importance = dict(zip(self.feature_names, self.model.feature_importances_))
            top_features = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5])
            
            # SHAP values for this prediction
            shap_dict = None
            if self.shap_explainer is not None:
                try:
                    shap_values = self.shap_explainer.shap_values(X_scaled)
                    if isinstance(shap_values, list):
                        # Multi-class: use predicted class
                        sv = shap_values[int(pred_class + 2)][0]
                    else:
                        sv = shap_values[0]
                    shap_dict = dict(zip(self.feature_names, sv))
                except:
                    pass
            
            explanation = self._generate_explanation(pred_class, top_features, latest, max_prob, shap_dict)
            
            pred_enum = {
                2: PredictionClass.STRONG_UP,
                1: PredictionClass.UP,
                0: PredictionClass.FLAT,
                -1: PredictionClass.DOWN,
                -2: PredictionClass.STRONG_DOWN,
            }[int(pred_class)]
            
            return PredictionResult(
                prediction=pred_enum,
                probability=float(max_prob),
                confidence=confidence,
                feature_importance=top_features,
                explanation=explanation,
                shap_values=shap_dict
            )
            
        except Exception as e:
            logger.debug(f"Prediction error: {e}")
            return None
    
    def _generate_explanation(
        self,
        prediction: int,
        top_features: Dict[str, float],
        features: pd.DataFrame,
        probability: float,
        shap_values: Optional[Dict[str, float]] = None
    ) -> str:
        """Generate explanation using SHAP or feature importance."""
        direction = {
            2: "STRONG UPWARD",
            1: "UPWARD",
            0: "SIDEWAYS",
            -1: "DOWNWARD",
            -2: "STRONG DOWNWARD"
        }[prediction]
        
        explanations = []
        
        # Use SHAP values if available for better explanations
        feature_impacts = shap_values if shap_values else top_features
        sorted_impacts = sorted(feature_impacts.items(), key=lambda x: abs(x[1]), reverse=True)
        
        for feat, impact in sorted_impacts[:3]:
            if abs(impact) < 0.01:
                continue
            
            val = features[feat].iloc[0] if feat in features.columns else 0
            impact_dir = "↑" if impact > 0 else "↓"
            
            # Human-readable feature names
            if 'rsi' in feat.lower():
                if val < 0.3:
                    explanations.append(f"RSI oversold ({impact_dir})")
                elif val > 0.7:
                    explanations.append(f"RSI overbought ({impact_dir})")
            elif 'macd' in feat.lower():
                explanations.append(f"MACD {'bullish' if val > 0 else 'bearish'} ({impact_dir})")
            elif 'momentum' in feat.lower() or 'roc' in feat.lower():
                explanations.append(f"Momentum {'positive' if val > 0 else 'negative'} ({impact_dir})")
            elif 'volume' in feat.lower():
                explanations.append(f"Volume {'high' if val > 1 else 'low'} ({impact_dir})")
            elif 'sma' in feat.lower() or 'ma_' in feat.lower():
                explanations.append(f"MA signal ({impact_dir})")
            elif 'adx' in feat.lower() or 'trend' in feat.lower():
                explanations.append(f"Trend {'strong' if val > 0.25 else 'weak'} ({impact_dir})")
        
        if not explanations:
            explanations.append("Based on overall pattern")
        
        base = f"Predicts {direction} ({probability:.0%}). "
        factors = "Factors: " + ", ".join(explanations[:3])
        
        return base + factors
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model."""
        if not self.is_trained or self.model is None:
            return {}
        return dict(sorted(
            zip(self.feature_names, self.model.feature_importances_),
            key=lambda x: x[1], reverse=True
        ))
    
    def save_model(self, path: str) -> None:
        """Save model to file."""
        if self.is_trained:
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'best_params': self.best_params,
                'metrics': self.training_metrics,
            }, path)
    
    def load_model(self, path: str) -> bool:
        """Load model from file."""
        try:
            data = joblib.load(path)
            self.model = data['model']
            self.scaler = data['scaler']
            self.feature_names = data['feature_names']
            self.best_params = data.get('best_params', {})
            self.training_metrics = data.get('metrics', {})
            self.is_trained = True
            return True
        except:
            return False


# Compatibility alias
MLPredictor = EnhancedMLPredictor


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    from .data_fetcher import DataFetcher
    from .indicators import TechnicalIndicators
    
    print("Testing Enhanced ML Predictor...")
    fetcher = DataFetcher()
    df = fetcher.fetch_ticker_data("AAPL", period="2y")
    
    if df is not None:
        ti = TechnicalIndicators(df)
        df_ind = ti.get_dataframe()
        
        predictor = EnhancedMLPredictor(optimize_hyperparams=True, use_shap=True)
        metrics = predictor.train(df_ind, n_optuna_trials=10)
        
        print(f"Training metrics: {metrics}")
        
        result = predictor.predict(df_ind)
        if result:
            print(f"Prediction: {result.prediction.name}")
            print(f"Confidence: {result.confidence} ({result.probability:.0%})")
            print(f"Explanation: {result.explanation}")
