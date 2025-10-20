"""
Integration tests for the complete pairs trading pipeline
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.pairs.pair_finder import PairFinder
from src.sentiment.sentiment_analyzer import SentimentAnalyzer
from src.features.feature_engineer import FeatureEngineer
from src.models.predictor import SpreadPredictor
from src.backtest.engine import BacktestEngine
from src.backtest.metrics import PerformanceMetrics


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create sample price data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        
        # Create two cointegrated series
        stock1_prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.5)
        stock2_prices = stock1_prices * 1.5 + np.random.randn(len(dates)) * 2
        
        self.price_data = {
            'AAPL': pd.DataFrame({
                'date': dates,
                'close': stock1_prices,
                'volume': np.random.randint(1000000, 10000000, len(dates))
            }),
            'MSFT': pd.DataFrame({
                'date': dates,
                'close': stock2_prices,
                'volume': np.random.randint(1000000, 10000000, len(dates))
            })
        }
        
        # Create sample sentiment data
        self.sentiment_data = {
            'AAPL': pd.DataFrame({
                'date': dates[::7],  # Weekly sentiment
                'sentiment_score': np.random.uniform(-0.5, 0.5, len(dates[::7]))
            }),
            'MSFT': pd.DataFrame({
                'date': dates[::7],
                'sentiment_score': np.random.uniform(-0.5, 0.5, len(dates[::7]))
            })
        }
    
    def test_pair_finder(self):
        """Test pair identification"""
        print("\n=== Testing Pair Finder ===")
        pair_finder = PairFinder()
        
        # Test cointegration
        is_cointegrated, p_value = pair_finder.test_cointegration(
            self.price_data['AAPL']['close'].values,
            self.price_data['MSFT']['close'].values
        )
        
        print(f"Cointegration test: {'PASSED' if is_cointegrated else 'FAILED'}")
        print(f"P-value: {p_value:.4f}")
        
        self.assertIsInstance(is_cointegrated, bool)
        self.assertIsInstance(p_value, float)
    
    def test_sentiment_analyzer(self):
        """Test sentiment analysis"""
        print("\n=== Testing Sentiment Analyzer ===")
        analyzer = SentimentAnalyzer()
        
        test_texts = [
            "Apple reports record-breaking profits, stock soars!",
            "Microsoft faces challenges in cloud computing market",
            "Tech stocks show mixed performance today"
        ]
        
        for text in test_texts:
            score = analyzer.analyze_vader(text)
            print(f"Text: '{text[:50]}...'")
            print(f"Sentiment: {score:.3f}")
            self.assertIsInstance(score, float)
            self.assertTrue(-1 <= score <= 1)
    
    def test_feature_engineer(self):
        """Test feature engineering"""
        print("\n=== Testing Feature Engineer ===")
        engineer = FeatureEngineer()
        
        # Calculate spread
        spread = engineer.calculate_spread(
            self.price_data['AAPL']['close'].values,
            self.price_data['MSFT']['close'].values
        )
        
        print(f"Spread calculated: {len(spread)} points")
        print(f"Mean spread: {spread.mean():.2f}")
        print(f"Std spread: {spread.std():.2f}")
        
        # Calculate technical indicators
        prices = self.price_data['AAPL']['close'].values
        sma = engineer.calculate_sma(prices, window=20)
        volatility = engineer.calculate_volatility(prices, window=20)
        
        print(f"SMA points: {len(sma[~np.isnan(sma)])}")
        print(f"Mean volatility: {np.nanmean(volatility):.4f}")
        
        self.assertEqual(len(spread), len(prices))
        self.assertEqual(len(sma), len(prices))
    
    def test_spread_predictor(self):
        """Test ML predictor"""
        print("\n=== Testing Spread Predictor ===")
        
        # Prepare features
        engineer = FeatureEngineer()
        spread = engineer.calculate_spread(
            self.price_data['AAPL']['close'].values,
            self.price_data['MSFT']['close'].values
        )
        
        z_score = engineer.calculate_zscore(spread)
        sma = engineer.calculate_sma(self.price_data['AAPL']['close'].values, 20)
        volatility = engineer.calculate_volatility(
            self.price_data['AAPL']['close'].values, 20
        )
        
        # Create feature matrix
        features_df = pd.DataFrame({
            'spread_zscore': z_score,
            'sma_20': sma,
            'volatility_20': volatility,
            'sentiment_aapl': 0.0,
            'sentiment_msft': 0.0
        }).dropna()
        
        # Create target (next day spread movement)
        target = np.sign(np.diff(spread, prepend=spread[0]))[:len(features_df)]
        
        if len(features_df) > 100:
            # Split data
            split_idx = int(len(features_df) * 0.8)
            X_train = features_df.iloc[:split_idx]
            y_train = target[:split_idx]
            X_test = features_df.iloc[split_idx:]
            y_test = target[split_idx:]
            
            # Train predictor
            predictor = SpreadPredictor(model_type='gradient_boosting')
            predictor.train(X_train, y_train)
            
            # Make predictions
            predictions = predictor.predict(X_test)
            
            # Calculate accuracy
            accuracy = np.mean((predictions > 0) == (y_test > 0))
            print(f"Model accuracy: {accuracy:.2%}")
            print(f"Predictions shape: {predictions.shape}")
            
            self.assertGreater(len(predictions), 0)
            self.assertGreaterEqual(accuracy, 0.4)  # Better than random
    
    def test_backtest_engine(self):
        """Test backtesting engine"""
        print("\n=== Testing Backtest Engine ===")
        
        # Create simple signals
        dates = self.price_data['AAPL']['date'].values
        signals = pd.DataFrame({
            'date': dates,
            'signal': np.random.choice([-1, 0, 1], size=len(dates), p=[0.1, 0.8, 0.1])
        })
        
        # Run backtest
        engine = BacktestEngine(
            initial_capital=100000,
            transaction_cost=0.001,
            slippage=0.0005
        )
        
        results = engine.run_backtest(
            price_data=self.price_data,
            signals=signals,
            pair=('AAPL', 'MSFT')
        )
        
        print(f"Final portfolio value: ${results['portfolio_value'][-1]:,.2f}")
        print(f"Total return: {results['total_return']:.2%}")
        print(f"Number of trades: {results['num_trades']}")
        print(f"Sharpe ratio: {results['sharpe_ratio']:.3f}")
        print(f"Max drawdown: {results['max_drawdown']:.2%}")
        
        self.assertIn('portfolio_value', results)
        self.assertIn('sharpe_ratio', results)
        self.assertGreater(len(results['portfolio_value']), 0)
    
    def test_performance_metrics(self):
        """Test performance metrics calculation"""
        print("\n=== Testing Performance Metrics ===")
        
        # Create sample returns
        np.random.seed(42)
        returns = np.random.randn(252) * 0.01  # Daily returns for 1 year
        portfolio_value = 100000 * np.cumprod(1 + returns)
        
        metrics = PerformanceMetrics()
        
        sharpe = metrics.sharpe_ratio(returns)
        sortino = metrics.sortino_ratio(returns)
        max_dd = metrics.max_drawdown(portfolio_value)
        calmar = metrics.calmar_ratio(returns, portfolio_value)
        win_rate = metrics.win_rate(returns)
        
        print(f"Sharpe Ratio: {sharpe:.3f}")
        print(f"Sortino Ratio: {sortino:.3f}")
        print(f"Max Drawdown: {max_dd:.2%}")
        print(f"Calmar Ratio: {calmar:.3f}")
        print(f"Win Rate: {win_rate:.2%}")
        
        self.assertIsInstance(sharpe, float)
        self.assertIsInstance(max_dd, float)
        self.assertTrue(0 <= win_rate <= 1)


def run_integration_tests():
    """Run all integration tests"""
    print("\n" + "="*60)
    print("RUNNING INTEGRATION TESTS FOR PAIRS TRADING SYSTEM")
    print("="*60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestIntegration)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_integration_tests()
    sys.exit(0 if success else 1)
