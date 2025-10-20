"""
Feature engineering for pairs trading models.
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from sklearn.preprocessing import StandardScaler


class FeatureEngineer:
    """Create features for ML models."""
    
    def __init__(self):
        """Initialize feature engineer."""
        self.scaler = StandardScaler()
        self.feature_names = []
    
    def create_spread_features(
        self,
        spread_df: pd.DataFrame,
        windows: List[int] = [5, 10, 20, 60]
    ) -> pd.DataFrame:
        """
        Create features from spread data.
        
        Args:
            spread_df: DataFrame with spread and zscore
            windows: Rolling window sizes
            
        Returns:
            DataFrame with features
        """
        df = spread_df.copy()
        
        # Current z-score (primary feature)
        df['zscore'] = df['zscore']
        
        # Spread features
        for window in windows:
            # Rolling mean
            df[f'spread_ma_{window}'] = df['spread'].rolling(window=window).mean()
            
            # Rolling std
            df[f'spread_std_{window}'] = df['spread'].rolling(window=window).std()
            
            # Z-score over different windows
            spread_mean = df['spread'].rolling(window=window).mean()
            spread_std = df['spread'].rolling(window=window).std()
            df[f'zscore_{window}'] = (df['spread'] - spread_mean) / spread_std
        
        # Momentum features
        df['spread_change'] = df['spread'].diff()
        df['spread_change_pct'] = df['spread'].pct_change()
        df['zscore_change'] = df['zscore'].diff()
        
        # Lagged features
        for lag in [1, 2, 3, 5]:
            df[f'zscore_lag_{lag}'] = df['zscore'].shift(lag)
            df[f'spread_lag_{lag}'] = df['spread'].shift(lag)
        
        return df
    
    def create_price_features(
        self,
        price_data: pd.DataFrame,
        ticker: str,
        windows: List[int] = [5, 10, 20, 50]
    ) -> pd.DataFrame:
        """
        Create technical features from price data.
        
        Args:
            price_data: DataFrame with price data
            ticker: Ticker symbol
            windows: Window sizes for indicators
            
        Returns:
            DataFrame with features
        """
        # Filter for ticker
        df = price_data[price_data['ticker'] == ticker].copy()
        df = df.sort_values('date')
        
        # Returns
        df['returns'] = df['adj_close'].pct_change()
        
        # Volatility
        for window in windows:
            df[f'volatility_{window}'] = df['returns'].rolling(window=window).std()
        
        # Moving averages
        for window in windows:
            df[f'sma_{window}'] = df['adj_close'].rolling(window=window).mean()
            df[f'price_to_sma_{window}'] = df['adj_close'] / df[f'sma_{window}']
        
        # RSI
        df['rsi_14'] = self._calculate_rsi(df['adj_close'], period=14)
        
        # Volume features
        if 'volume' in df.columns:
            df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma_20']
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index.
        
        Args:
            prices: Price series
            period: RSI period
            
        Returns:
            RSI values
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def merge_features(
        self,
        spread_features: pd.DataFrame,
        price_features_1: pd.DataFrame,
        price_features_2: pd.DataFrame,
        sentiment_features_1: pd.DataFrame = None,
        sentiment_features_2: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Merge all features into single DataFrame.
        
        Args:
            spread_features: Spread-based features
            price_features_1: Features for stock 1
            price_features_2: Features for stock 2
            sentiment_features_1: Sentiment features for stock 1
            sentiment_features_2: Sentiment features for stock 2
            
        Returns:
            Merged feature DataFrame
        """
        # Start with spread features
        df = spread_features.copy()
        
        # Merge price features for stock 1
        price_cols_1 = [c for c in price_features_1.columns if c not in ['ticker', 'date']]
        price_1_renamed = price_features_1[['date'] + price_cols_1].copy()
        price_1_renamed.columns = ['date'] + [f'{c}_stock1' for c in price_cols_1]
        df = df.merge(price_1_renamed, on='date', how='left')
        
        # Merge price features for stock 2
        price_cols_2 = [c for c in price_features_2.columns if c not in ['ticker', 'date']]
        price_2_renamed = price_features_2[['date'] + price_cols_2].copy()
        price_2_renamed.columns = ['date'] + [f'{c}_stock2' for c in price_cols_2]
        df = df.merge(price_2_renamed, on='date', how='left')
        
        # Merge sentiment features if provided
        if sentiment_features_1 is not None:
            sent_cols_1 = [c for c in sentiment_features_1.columns if c not in ['ticker', 'date']]
            sent_1_renamed = sentiment_features_1[['date'] + sent_cols_1].copy()
            sent_1_renamed.columns = ['date'] + [f'{c}_stock1' for c in sent_cols_1]
            df = df.merge(sent_1_renamed, on='date', how='left')
        
        if sentiment_features_2 is not None:
            sent_cols_2 = [c for c in sentiment_features_2.columns if c not in ['ticker', 'date']]
            sent_2_renamed = sentiment_features_2[['date'] + sent_cols_2].copy()
            sent_2_renamed.columns = ['date'] + [f'{c}_stock2' for c in sent_cols_2]
            df = df.merge(sent_2_renamed, on='date', how='left')
        
        # Relative features (difference between stocks)
        if 'returns_stock1' in df.columns and 'returns_stock2' in df.columns:
            df['returns_diff'] = df['returns_stock1'] - df['returns_stock2']
        
        if 'volatility_20_stock1' in df.columns and 'volatility_20_stock2' in df.columns:
            df['volatility_ratio'] = df['volatility_20_stock1'] / df['volatility_20_stock2']
        
        return df
    
    def prepare_ml_features(
        self,
        feature_df: pd.DataFrame,
        target_column: str = 'future_zscore',
        drop_na: bool = True
    ) -> tuple:
        """
        Prepare features for machine learning.
        
        Args:
            feature_df: DataFrame with all features
            target_column: Name of target variable
            drop_na: Whether to drop NaN values
            
        Returns:
            (X, y, feature_names)
        """
        df = feature_df.copy()
        
        # Create target (future z-score for regression)
        if target_column not in df.columns:
            df['future_zscore'] = df['zscore'].shift(-1)
        
        # Drop non-feature columns
        exclude_cols = ['date', 'pair_id', 'ticker', 'ticker_1', 'ticker_2', target_column]
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        if drop_na:
            df = df.dropna(subset=feature_cols + [target_column])
        
        X = df[feature_cols].values
        y = df[target_column].values
        
        self.feature_names = feature_cols
        
        return X, y, feature_cols
    
    def scale_features(
        self,
        X: np.ndarray,
        fit: bool = True
    ) -> np.ndarray:
        """
        Scale features using StandardScaler.
        
        Args:
            X: Feature array
            fit: Whether to fit the scaler
            
        Returns:
            Scaled features
        """
        if fit:
            return self.scaler.fit_transform(X)
        else:
            return self.scaler.transform(X)


def create_target_labels(
    spread_df: pd.DataFrame,
    entry_threshold: float = 2.0,
    exit_threshold: float = 0.5
) -> pd.DataFrame:
    """
    Create target labels for classification.
    
    Args:
        spread_df: DataFrame with z-scores
        entry_threshold: Z-score threshold for entry
        exit_threshold: Z-score threshold for exit
        
    Returns:
        DataFrame with labels
    """
    df = spread_df.copy()
    
    # Future z-score
    df['future_zscore'] = df['zscore'].shift(-1)
    
    # Label based on current and future z-score
    df['signal'] = 'HOLD'
    
    # Long signal: z-score is below -threshold and will revert up
    df.loc[
        (df['zscore'] < -entry_threshold) & (df['future_zscore'] > df['zscore']),
        'signal'
    ] = 'LONG'
    
    # Short signal: z-score is above +threshold and will revert down
    df.loc[
        (df['zscore'] > entry_threshold) & (df['future_zscore'] < df['zscore']),
        'signal'
    ] = 'SHORT'
    
    # Exit signal: z-score is near zero
    df.loc[
        abs(df['zscore']) < exit_threshold,
        'signal'
    ] = 'EXIT'
    
    return df


if __name__ == "__main__":
    print("Feature Engineering Module")
    print("-" * 50)
    
    engineer = FeatureEngineer()
    print("✓ Initialized FeatureEngineer")
