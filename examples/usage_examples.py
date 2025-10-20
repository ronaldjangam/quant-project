"""
Example script showing how to use individual modules.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Example 1: Fetch stock data
def example_fetch_data():
    """Fetch stock price data."""
    from src.data.data_loader import StockDataLoader
    
    print("\n" + "="*60)
    print("EXAMPLE 1: Fetching Stock Data")
    print("="*60)
    
    loader = StockDataLoader(data_source='yfinance')
    
    # Fetch data for a few tickers
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN']
    start_date = '2023-01-01'
    end_date = '2024-01-01'
    
    df = loader.fetch_multiple_tickers(tickers, start_date, end_date)
    
    print(f"\nFetched {len(df)} records")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"\nSample data:")
    print(df.head())
    
    return df


# Example 2: Find pairs
def example_find_pairs(price_data):
    """Find cointegrated pairs."""
    from src.pairs.pair_finder import CointegrationPairFinder
    
    print("\n" + "="*60)
    print("EXAMPLE 2: Finding Cointegrated Pairs")
    print("="*60)
    
    finder = CointegrationPairFinder(p_value_threshold=0.05)
    pairs = finder.find_pairs(price_data)
    
    print(f"\nFound {len(pairs)} cointegrated pairs")
    
    if pairs:
        pairs_df = pd.DataFrame(pairs)
        print("\nTop 5 pairs:")
        print(pairs_df.head()[['ticker_1', 'ticker_2', 'p_value', 'correlation']])
    
    return pairs


# Example 3: Sentiment analysis
def example_sentiment_analysis():
    """Analyze sentiment of text."""
    from src.sentiment.sentiment_analyzer import SentimentAnalyzer
    
    print("\n" + "="*60)
    print("EXAMPLE 3: Sentiment Analysis")
    print("="*60)
    
    analyzer = SentimentAnalyzer(model_type='vader')
    
    texts = [
        "Apple stock surges to all-time high on record iPhone sales",
        "Tech stocks plummet as recession fears intensify",
        "Microsoft announces steady quarterly earnings, meeting expectations",
        "Amazon faces regulatory scrutiny over market practices"
    ]
    
    print("\nAnalyzing sentiment of news headlines:\n")
    
    for text in texts:
        sentiment = analyzer.analyze_text(text)
        
        print(f"Text: {text}")
        print(f"  Compound: {sentiment['compound']:>6.3f} | ", end="")
        print(f"Pos: {sentiment['positive']:.3f} | ", end="")
        print(f"Neg: {sentiment['negative']:.3f}\n")


# Example 4: Feature engineering
def example_feature_engineering(price_data, pair):
    """Create features for ML."""
    from src.features.feature_engineer import FeatureEngineer
    from src.pairs.pair_finder import calculate_spread
    
    print("\n" + "="*60)
    print("EXAMPLE 4: Feature Engineering")
    print("="*60)
    
    if not pair:
        print("No pairs available")
        return
    
    # Calculate spread
    spread_df = calculate_spread(
        price_data,
        pair['ticker_1'],
        pair['ticker_2'],
        pair['hedge_ratio']
    )
    
    print(f"\nPair: {pair['ticker_1']} / {pair['ticker_2']}")
    print(f"Spread observations: {len(spread_df)}")
    
    # Engineer features
    engineer = FeatureEngineer()
    features_df = engineer.create_spread_features(spread_df)
    
    print(f"Created {len(features_df.columns)} features")
    print(f"\nFeature columns: {list(features_df.columns[:10])}...")
    
    return features_df


# Example 5: Train a simple model
def example_train_model(features_df):
    """Train ML model."""
    from src.models.predictor import ModelTrainer
    from src.features.feature_engineer import FeatureEngineer
    
    print("\n" + "="*60)
    print("EXAMPLE 5: Training ML Model")
    print("="*60)
    
    if features_df is None or len(features_df) < 100:
        print("Insufficient data for training")
        return
    
    # Prepare data
    engineer = FeatureEngineer()
    X, y, feature_names = engineer.prepare_ml_features(features_df)
    
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Split data
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train model
    trainer = ModelTrainer(
        model_type='gradient_boosting',
        n_estimators=100,
        learning_rate=0.05
    )
    
    trainer.model.train(X_train, y_train, X_test, y_test)
    
    # Evaluate
    metrics = trainer.evaluate(X_test, y_test)
    print(f"\nModel Performance:")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  R²:   {metrics['r2']:.4f}")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("AI-POWERED PAIRS TRADING - USAGE EXAMPLES")
    print("="*60)
    
    # Example 1: Fetch data
    price_data = example_fetch_data()
    
    # Example 2: Find pairs
    pairs = example_find_pairs(price_data)
    
    # Example 3: Sentiment analysis
    example_sentiment_analysis()
    
    # Example 4: Feature engineering
    if pairs:
        features_df = example_feature_engineering(price_data, pairs[0])
        
        # Example 5: Train model
        if features_df is not None:
            example_train_model(features_df)
    
    print("\n" + "="*60)
    print("✅ All examples completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
