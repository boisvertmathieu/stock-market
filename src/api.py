"""
API Module
JSON API layer for communication with TUI via stdin/stdout.
"""

import sys
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

from .data_fetcher import DataFetcher, TOP_100_TICKERS
from .indicators import TechnicalIndicators
from .predictor import MLPredictor
from .scanner import MarketScanner, TickerAnalysis
from .backtester import Backtester

logger = logging.getLogger(__name__)


class StockAPI:
    """
    JSON API for stock analysis.
    Receives commands via stdin, outputs results via stdout.
    """
    
    def __init__(self):
        self.fetcher = DataFetcher()
        self.scanner = MarketScanner()
        self._models: Dict[str, MLPredictor] = {}
    
    def _get_predictor(self, ticker: str) -> MLPredictor:
        """Get or create predictor for ticker."""
        if ticker not in self._models:
            self._models[ticker] = MLPredictor()
        return self._models[ticker]
    
    def handle_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle incoming API command.
        
        Args:
            command: Dictionary with 'action' and optional 'params'
            
        Returns:
            Response dictionary
        """
        action = command.get('action', '')
        params = command.get('params', {})
        
        handlers = {
            'ping': self._handle_ping,
            'get_tickers': self._handle_get_tickers,
            'analyze': self._handle_analyze,
            'scan': self._handle_scan,
            'quick_scan': self._handle_quick_scan,
            'backtest': self._handle_backtest,
            'get_price': self._handle_get_price,
            'get_market_summary': self._handle_market_summary,
        }
        
        handler = handlers.get(action)
        if handler:
            try:
                return handler(params)
            except Exception as e:
                logger.error(f"Error handling {action}: {e}")
                return {'error': str(e), 'action': action}
        else:
            return {'error': f'Unknown action: {action}'}
    
    def _handle_ping(self, params: Dict) -> Dict[str, Any]:
        """Health check."""
        return {
            'status': 'ok',
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0'
        }
    
    def _handle_get_tickers(self, params: Dict) -> Dict[str, Any]:
        """Get list of available tickers."""
        return {
            'tickers': TOP_100_TICKERS,
            'count': len(TOP_100_TICKERS)
        }
    
    def _handle_analyze(self, params: Dict) -> Dict[str, Any]:
        """Analyze a single ticker in detail."""
        ticker = params.get('ticker', '').upper()
        if not ticker:
            return {'error': 'Missing ticker parameter'}
        
        period = params.get('period', '1y')
        
        # Fetch data
        df = self.fetcher.fetch_ticker_data(ticker, period=period)
        if df is None:
            return {'error': f'Could not fetch data for {ticker}'}
        
        # Get info
        info = self.fetcher.get_ticker_info(ticker)
        
        # Technical analysis
        ti = TechnicalIndicators(df)
        analysis = ti.get_comprehensive_analysis()
        
        # ML prediction
        predictor = self._get_predictor(ticker)
        df_with_indicators = ti.get_dataframe()
        
        if not predictor.is_trained:
            predictor.train(df_with_indicators, n_splits=3)
        
        ml_result = predictor.predict(df_with_indicators)
        
        # Build response
        response = {
            'ticker': ticker,
            'info': info,
            'price': analysis['price'],
            'indicators': analysis['indicators'],
            'technical_signal': analysis['overall'],
            'ml_prediction': {
                'prediction': ml_result.prediction.name if ml_result else None,
                'probability': ml_result.probability if ml_result else 0,
                'confidence': ml_result.confidence if ml_result else 'N/A',
                'explanation': ml_result.explanation if ml_result else '',
                'feature_importance': ml_result.feature_importance if ml_result else {},
            },
            'training_metrics': predictor.training_metrics,
            'historical_data': {
                'dates': df.index.strftime('%Y-%m-%d').tolist()[-30:],
                'close': df['Close'].round(2).tolist()[-30:],
                'volume': df['Volume'].astype(int).tolist()[-30:],
            }
        }
        
        # Calculate suggested levels
        current_price = analysis['price']['current']
        atr = analysis['raw_indicators'].get('ATR', current_price * 0.02)
        
        if analysis['overall']['signal_value'] > 0:
            response['trade_suggestion'] = {
                'action': 'BUY',
                'entry': round(current_price, 2),
                'stop_loss': round(current_price - 2 * atr, 2),
                'take_profit': round(current_price + 3 * atr, 2),
            }
        elif analysis['overall']['signal_value'] < 0:
            response['trade_suggestion'] = {
                'action': 'SELL',
                'entry': round(current_price, 2),
                'stop_loss': round(current_price + 2 * atr, 2),
                'take_profit': round(current_price - 3 * atr, 2),
            }
        else:
            response['trade_suggestion'] = {'action': 'HOLD'}
        
        return response
    
    def _handle_scan(self, params: Dict) -> Dict[str, Any]:
        """Full market scan (all 100 tickers)."""
        tickers = params.get('tickers', TOP_100_TICKERS)
        period = params.get('period', '1y')
        
        results = self.scanner.scan_tickers(tickers=tickers, period=period)
        
        return {
            'results': [self._analysis_to_dict(r) for r in results],
            'summary': self.scanner.get_market_summary(results),
            'timestamp': datetime.now().isoformat()
        }
    
    def _handle_quick_scan(self, params: Dict) -> Dict[str, Any]:
        """Quick scan of top 20 tickers."""
        tickers = params.get('tickers', TOP_100_TICKERS[:20])
        period = params.get('period', '1y')
        
        results = self.scanner.scan_tickers(tickers=tickers, period=period)
        
        return {
            'results': [self._analysis_to_dict(r) for r in results],
            'top_buys': [self._analysis_to_dict(r) for r in self.scanner.get_top_opportunities(results, 'BUY', 5)],
            'top_sells': [self._analysis_to_dict(r) for r in self.scanner.get_top_opportunities(results, 'SELL', 5)],
            'timestamp': datetime.now().isoformat()
        }
    
    def _handle_backtest(self, params: Dict) -> Dict[str, Any]:
        """Run backtest for a ticker."""
        ticker = params.get('ticker', '').upper()
        if not ticker:
            return {'error': 'Missing ticker parameter'}
        
        period = params.get('period', '2y')
        initial_capital = params.get('capital', 100000)
        
        # Fetch and prepare data
        df = self.fetcher.fetch_ticker_data(ticker, period=period)
        if df is None:
            return {'error': f'Could not fetch data for {ticker}'}
        
        # Calculate indicators
        ti = TechnicalIndicators(df)
        df_with_indicators = ti.get_dataframe()
        
        # Train predictor
        predictor = self._get_predictor(ticker)
        predictor.train(df_with_indicators, n_splits=5)
        
        # Generate signals using technical analysis (simplified for speed)
        signals = df_with_indicators['RSI'].apply(
            lambda x: 1 if x < 30 else (-1 if x > 70 else 0)
        ).fillna(0)
        
        # Run backtest
        backtester = Backtester(initial_capital=initial_capital)
        result = backtester.run(df, signals)
        
        return {
            'ticker': ticker,
            'period': period,
            'metrics': {
                'total_return': result.total_return,
                'annualized_return': result.annualized_return,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'win_rate': result.win_rate,
                'profit_factor': result.profit_factor,
                'total_trades': result.total_trades,
                'avg_trade_pnl': result.avg_trade_pnl,
            },
            'equity_curve': {
                'dates': result.equity_curve.index.strftime('%Y-%m-%d').tolist()[-60:],
                'values': result.equity_curve.round(2).tolist()[-60:],
            },
            'monthly_returns': result.monthly_returns.round(4).to_dict() if len(result.monthly_returns) > 0 else {},
        }
    
    def _handle_get_price(self, params: Dict) -> Dict[str, Any]:
        """Get current price for a ticker."""
        ticker = params.get('ticker', '').upper()
        if not ticker:
            return {'error': 'Missing ticker parameter'}
        
        price_info = self.fetcher.get_current_price(ticker)
        if price_info:
            return {'ticker': ticker, **price_info}
        else:
            return {'error': f'Could not get price for {ticker}'}
    
    def _handle_market_summary(self, params: Dict) -> Dict[str, Any]:
        """Get cached market summary."""
        if self.scanner.results:
            results = list(self.scanner.results.values())
            return self.scanner.get_market_summary(results)
        else:
            return {'error': 'No scan results available. Run scan first.'}
    
    def _analysis_to_dict(self, analysis: TickerAnalysis) -> Dict[str, Any]:
        """Convert TickerAnalysis to dictionary."""
        return {
            'ticker': analysis.ticker,
            'name': analysis.name,
            'sector': analysis.sector,
            'price': analysis.price,
            'change': analysis.change,
            'change_percent': analysis.change_percent,
            'volume_ratio': analysis.volume_ratio,
            'technical_signal': analysis.technical_signal,
            'technical_score': analysis.technical_score,
            'ml_prediction': analysis.ml_prediction,
            'ml_confidence': analysis.ml_confidence,
            'ml_probability': analysis.ml_probability,
            'overall_action': analysis.overall_action,
            'action_strength': analysis.action_strength,
            'key_factors': analysis.key_factors,
            'entry_price': analysis.entry_price,
            'stop_loss': analysis.stop_loss,
            'take_profit': analysis.take_profit,
            'risk_reward_ratio': analysis.risk_reward_ratio,
        }
    
    def run_interactive(self):
        """Run in interactive mode, reading JSON commands from stdin."""
        print(json.dumps({'status': 'ready', 'message': 'Stock API ready'}), flush=True)
        
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            
            try:
                command = json.loads(line)
                response = self.handle_command(command)
                print(json.dumps(response), flush=True)
            except json.JSONDecodeError as e:
                print(json.dumps({'error': f'Invalid JSON: {e}'}), flush=True)
            except Exception as e:
                print(json.dumps({'error': str(e)}), flush=True)


if __name__ == "__main__":
    api = StockAPI()
    api.run_interactive()
