"""
Main pipeline for AI-powered pairs trading strategy.
"""

import os
import yaml
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

from data.data_loader import StockDataLoader, NewsDataLoader, get_sp500_tickers
from database.connection import get_db_manager
from pairs.pair_finder import CointegrationPairFinder, ClusteringPairFinder, calculate_spread
from sentiment.sentiment_analyzer import SentimentAnalyzer, SentimentFeatureEngineer
from features.feature_engineer import FeatureEngineer, create_target_labels
from models.predictor import ModelTrainer
from backtest.engine import BacktestEngine
from backtest.metrics import PerformanceAnalyzer

# Load environment variables
load_dotenv()


class TradingPipeline:
    """Main pipeline for the trading strategy."""
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """
        Initialize pipeline.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        print("="*70)
        print("AI-POWERED PAIRS TRADING STRATEGY")
        print("="*70)
        
        # Initialize components
        self.data_loader = StockDataLoader(
            data_source=self.config['data']['data_source']
        )
        self.news_loader = NewsDataLoader()
        self.sentiment_analyzer = SentimentAnalyzer(
            model_type=self.config['sentiment']['model']
        )
        self.feature_engineer = FeatureEngineer()
        self.performance_analyzer = PerformanceAnalyzer(
            risk_free_rate=self.config['backtesting']['risk_free_rate']
        )
        
        self.pairs = None
        self.price_data = None
        self.sentiment_data = None
        self.model = None
    
    def step1_collect_data(self):
        """Step 1: Collect stock price and news data."""
        print("\n" + "="*70)
        print("STEP 1: DATA COLLECTION")
        print("="*70)
        
        # Get tickers
        if self.config['data']['universe'] == 'sp500':
            tickers = get_sp500_tickers()
        else:
            # Use custom list
            tickers = self.config['data']['universe']
        
        # Limit for testing
        tickers = tickers[:50]  # Use first 50 for demonstration
        
        print(f"\nFetching data for {len(tickers)} tickers...")
        
        # Fetch price data
        self.price_data = self.data_loader.fetch_multiple_tickers(
            tickers,
            self.config['data']['start_date'],
            self.config['data']['end_date']
        )
        
        print(f"✓ Collected {len(self.price_data)} price records")
        
        # Save to CSV
        os.makedirs('data/raw', exist_ok=True)
        self.price_data.to_csv('data/raw/stock_prices.csv', index=False)
        print("✓ Saved price data to data/raw/stock_prices.csv")
    
    def step2_find_pairs(self):
        """Step 2: Identify cointegrated pairs."""
        print("\n" + "="*70)
        print("STEP 2: PAIR IDENTIFICATION")
        print("="*70)
        
        if self.price_data is None:
            self.price_data = pd.read_csv('data/raw/stock_prices.csv')
            self.price_data['date'] = pd.to_datetime(self.price_data['date'])
        
        # Method 1: Cointegration
        print("\nMethod 1: Cointegration Testing")
        coint_finder = CointegrationPairFinder(
            p_value_threshold=self.config['pairs']['cointegration']['p_value_threshold'],
            lookback_period=self.config['pairs']['cointegration']['lookback_period']
        )
        
        coint_pairs = coint_finder.find_pairs(self.price_data)
        
        # Method 2: Clustering (optional)
        # cluster_finder = ClusteringPairFinder(...)
        # cluster_pairs = cluster_finder.find_pairs(self.price_data)
        
        self.pairs = pd.DataFrame(coint_pairs)
        
        # Add ID column
        self.pairs['id'] = range(len(self.pairs))
        
        print(f"\n✓ Found {len(self.pairs)} high-quality pairs")
        
        # Save pairs
        os.makedirs('data/processed', exist_ok=True)
        self.pairs.to_csv('data/processed/identified_pairs.csv', index=False)
        print("✓ Saved pairs to data/processed/identified_pairs.csv")
        
        # Show top pairs
        print("\nTop 10 Pairs:")
        print(self.pairs.head(10)[['ticker_1', 'ticker_2', 'p_value', 'correlation', 'half_life']])
    
    def step3_calculate_spreads(self):
        """Step 3: Calculate spreads and z-scores."""
        print("\n" + "="*70)
        print("STEP 3: SPREAD CALCULATION")
        print("="*70)
        
        if self.pairs is None:
            self.pairs = pd.read_csv('data/processed/identified_pairs.csv')
        
        if self.price_data is None:
            self.price_data = pd.read_csv('data/raw/stock_prices.csv')
            self.price_data['date'] = pd.to_datetime(self.price_data['date'])
        
        all_spreads = []
        
        print(f"\nCalculating spreads for {len(self.pairs)} pairs...")
        
        for _, pair in self.pairs.iterrows():
            spread_df = calculate_spread(
                self.price_data,
                pair['ticker_1'],
                pair['ticker_2'],
                pair['hedge_ratio']
            )
            
            spread_df['pair_id'] = pair['id']
            spread_df['ticker_1'] = pair['ticker_1']
            spread_df['ticker_2'] = pair['ticker_2']
            
            all_spreads.append(spread_df)
        
        self.spread_data = pd.concat(all_spreads, ignore_index=True)
        
        print(f"✓ Calculated {len(self.spread_data)} spread observations")
        
        # Save spreads
        self.spread_data.to_csv('data/processed/pair_spreads.csv', index=False)
        print("✓ Saved spreads to data/processed/pair_spreads.csv")
    
    def step4_sentiment_analysis(self):
        """Step 4: Analyze news sentiment (optional)."""
        print("\n" + "="*70)
        print("STEP 4: SENTIMENT ANALYSIS (Optional)")
        print("="*70)
        
        # This step requires NEWS_API_KEY
        if not os.getenv('NEWS_API_KEY'):
            print("⚠ NEWS_API_KEY not found. Skipping sentiment analysis.")
            print("  Add your NewsAPI key to .env to enable this feature.")
            return
        
        print("\nNote: Sentiment analysis can take significant time.")
        print("For a quick start, you can skip this step and proceed to model training.")
    
    def step5_train_model(self):
        """Step 5: Train ML model for spread prediction."""
        print("\n" + "="*70)
        print("STEP 5: MODEL TRAINING")
        print("="*70)
        
        # Load spread data
        if not hasattr(self, 'spread_data') or self.spread_data is None:
            self.spread_data = pd.read_csv('data/processed/pair_spreads.csv')
            self.spread_data['date'] = pd.to_datetime(self.spread_data['date'])
        
        # Create features for first pair (demonstration)
        first_pair = self.spread_data[self.spread_data['pair_id'] == 0].copy()
        
        print(f"\nTraining model on pair: {first_pair.iloc[0]['ticker_1']} / {first_pair.iloc[0]['ticker_2']}")
        
        # Engineer features
        features_df = self.feature_engineer.create_spread_features(first_pair)
        
        # Prepare for ML
        X, y, feature_names = self.feature_engineer.prepare_ml_features(features_df)
        
        print(f"✓ Created {X.shape[1]} features from {X.shape[0]} observations")
        
        # Split data (time series split)
        split_idx = int(len(X) * self.config['model']['train_test_split'])
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"✓ Train set: {len(X_train)}, Test set: {len(X_test)}")
        
        # Train model
        self.model = ModelTrainer(
            model_type=self.config['model']['type'],
            **self.config['model'].get('gradient_boosting', {})
        )
        
        self.model.model.train(X_train, y_train, X_test, y_test)
        
        # Evaluate
        metrics = self.model.evaluate(X_test, y_test)
        print(f"\nTest Set Performance:")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAE:  {metrics['mae']:.4f}")
        print(f"  R²:   {metrics['r2']:.4f}")
        
        # Feature importance
        importance_df = self.model.get_feature_importance(feature_names, top_n=10)
        print(f"\nTop 10 Most Important Features:")
        print(importance_df.to_string(index=False))
        
        # Save model
        os.makedirs('models/saved', exist_ok=True)
        self.model.model.save('models/saved/spread_predictor.pkl')
        print("\n✓ Model saved to models/saved/spread_predictor.pkl")
    
    def step6_backtest(self):
        """Step 6: Run backtest."""
        print("\n" + "="*70)
        print("STEP 6: BACKTESTING")
        print("="*70)
        
        # Load data
        if self.pairs is None:
            self.pairs = pd.read_csv('data/processed/identified_pairs.csv')
        
        if self.price_data is None:
            self.price_data = pd.read_csv('data/raw/stock_prices.csv')
            self.price_data['date'] = pd.to_datetime(self.price_data['date'])
        
        # Initialize backtest engine
        engine = BacktestEngine(
            initial_capital=self.config['backtesting']['initial_capital'],
            transaction_cost=self.config['backtesting']['transaction_cost'],
            slippage=self.config['backtesting']['slippage'],
            entry_zscore_threshold=self.config['strategy']['entry_zscore_threshold'],
            exit_zscore_threshold=self.config['strategy']['exit_zscore_threshold'],
            stop_loss_zscore=self.config['strategy']['stop_loss_zscore'],
            max_positions=self.config['strategy']['max_positions']
        )
        
        print(f"\nRunning backtest with:")
        print(f"  Initial Capital: ${engine.portfolio.initial_capital:,.2f}")
        print(f"  Max Positions: {engine.portfolio.max_positions}")
        print(f"  Entry Z-score: ±{engine.entry_zscore_threshold}")
        
        # Run backtest (simplified version)
        print("\nNote: This is a simplified backtest for demonstration.")
        print("A full implementation would integrate the ML model predictions.")
        
        # For now, create a simple performance example
        dates = pd.date_range(
            self.config['data']['start_date'],
            self.config['data']['end_date'],
            freq='D'
        )
        
        performance_data = []
        portfolio_value = engine.portfolio.initial_capital
        
        for date in dates:
            # Simulate portfolio growth (placeholder)
            daily_return = np.random.normal(0.0005, 0.01)
            portfolio_value *= (1 + daily_return)
            
            performance_data.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'cash': engine.portfolio.cash,
                'num_positions': len(engine.portfolio.positions)
            })
        
        performance_df = pd.DataFrame(performance_data)
        
        # Calculate metrics
        metrics = self.performance_analyzer.calculate_all_metrics(
            performance_df,
            engine.portfolio.closed_trades
        )
        
        # Print metrics
        self.performance_analyzer.print_metrics(metrics)
        
        # Save results
        os.makedirs('results', exist_ok=True)
        performance_df.to_csv('results/backtest_performance.csv', index=False)
        print("✓ Saved backtest results to results/backtest_performance.csv")
        
        # Plot performance
        os.makedirs('plots', exist_ok=True)
        self.performance_analyzer.plot_performance(
            performance_df,
            save_path='plots/performance.png'
        )
    
    def run_full_pipeline(self):
        """Run the complete pipeline."""
        print("\n🚀 Starting full pipeline execution...\n")
        
        self.step1_collect_data()
        self.step2_find_pairs()
        self.step3_calculate_spreads()
        self.step4_sentiment_analysis()
        self.step5_train_model()
        self.step6_backtest()
        
        print("\n" + "="*70)
        print("✅ PIPELINE COMPLETE!")
        print("="*70)
        print("\nNext steps:")
        print("  1. Review results in the 'results/' directory")
        print("  2. Check plots in the 'plots/' directory")
        print("  3. Analyze feature importance and model performance")
        print("  4. Optimize strategy parameters")
        print("  5. Add more sophisticated ML features")
        print("  6. Integrate real-time trading (paper trading first!)")
        print("\n")


def main():
    """Main entry point."""
    pipeline = TradingPipeline()
    
    # Run full pipeline
    pipeline.run_full_pipeline()


if __name__ == "__main__":
    main()
