"""
Basic tests for the trading system.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def test_cointegration_pair_finder():
    """Test cointegration pair finder."""
    from src.pairs.pair_finder import CointegrationPairFinder
    
    finder = CointegrationPairFinder(p_value_threshold=0.05)
    assert finder.p_value_threshold == 0.05
    assert finder.lookback_period == 252


def test_sentiment_analyzer():
    """Test sentiment analyzer."""
    from src.sentiment.sentiment_analyzer import SentimentAnalyzer
    
    analyzer = SentimentAnalyzer(model_type='vader')
    
    # Test positive sentiment
    sentiment = analyzer.analyze_text("Amazing earnings! Stock surges to record high!")
    assert sentiment['compound'] > 0
    assert sentiment['positive'] > sentiment['negative']
    
    # Test negative sentiment
    sentiment = analyzer.analyze_text("Terrible results. Company faces bankruptcy.")
    assert sentiment['compound'] < 0
    assert sentiment['negative'] > sentiment['positive']


def test_feature_engineer():
    """Test feature engineering."""
    from src.features.feature_engineer import FeatureEngineer
    
    engineer = FeatureEngineer()
    
    # Create dummy spread data
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    spread_df = pd.DataFrame({
        'date': dates,
        'spread': np.random.randn(len(dates)),
        'zscore': np.random.randn(len(dates)),
        'mean': np.random.randn(len(dates)),
        'std_dev': np.abs(np.random.randn(len(dates)))
    })
    
    # Create features
    features_df = engineer.create_spread_features(spread_df)
    
    assert len(features_df) > 0
    assert 'zscore' in features_df.columns


def test_backtest_portfolio():
    """Test portfolio initialization."""
    from src.backtest.engine import Portfolio
    
    portfolio = Portfolio(
        initial_capital=100000,
        max_positions=10
    )
    
    assert portfolio.initial_capital == 100000
    assert portfolio.cash == 100000
    assert len(portfolio.positions) == 0
    assert portfolio.can_open_position() == True


def test_performance_analyzer():
    """Test performance metrics calculation."""
    from src.backtest.metrics import PerformanceAnalyzer
    
    analyzer = PerformanceAnalyzer(risk_free_rate=0.02)
    
    # Test Sharpe ratio calculation
    returns = pd.Series(np.random.randn(252) * 0.01)
    sharpe = analyzer.calculate_sharpe_ratio(returns)
    
    assert isinstance(sharpe, float)
    
    # Test max drawdown
    cumulative_returns = pd.Series(np.cumsum(returns))
    max_dd = analyzer.calculate_max_drawdown(cumulative_returns)
    
    assert max_dd >= 0


def test_data_loader():
    """Test data loader initialization."""
    from src.data.data_loader import StockDataLoader
    
    loader = StockDataLoader(data_source='yfinance')
    assert loader.data_source == 'yfinance'


def test_spread_calculation():
    """Test spread calculation."""
    from src.pairs.pair_finder import calculate_spread
    
    # Create dummy price data
    dates = pd.date_range('2023-01-01', '2023-12-31', freq='D')
    price_data = pd.DataFrame({
        'date': list(dates) * 2,
        'ticker': ['AAPL'] * len(dates) + ['MSFT'] * len(dates),
        'adj_close': list(np.random.randn(len(dates)) * 10 + 150) * 2
    })
    
    spread_df = calculate_spread(price_data, 'AAPL', 'MSFT', hedge_ratio=1.0)
    
    assert 'spread' in spread_df.columns
    assert 'zscore' in spread_df.columns
    assert len(spread_df) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
