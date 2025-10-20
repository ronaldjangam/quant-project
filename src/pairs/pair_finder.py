"""
Pair identification using cointegration tests and clustering.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from itertools import combinations
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


class CointegrationPairFinder:
    """Find cointegrated stock pairs using statistical tests."""
    
    def __init__(
        self,
        p_value_threshold: float = 0.05,
        lookback_period: int = 252
    ):
        """
        Initialize pair finder.
        
        Args:
            p_value_threshold: Maximum p-value for cointegration
            lookback_period: Number of trading days to analyze
        """
        self.p_value_threshold = p_value_threshold
        self.lookback_period = lookback_period
    
    def test_cointegration(
        self,
        series1: pd.Series,
        series2: pd.Series
    ) -> Tuple[float, float, float]:
        """
        Test if two price series are cointegrated.
        
        Args:
            series1: First price series
            series2: Second price series
            
        Returns:
            (p_value, hedge_ratio, half_life)
        """
        # Ensure series are aligned
        df = pd.DataFrame({'s1': series1, 's2': series2}).dropna()
        
        if len(df) < 30:
            return (1.0, 0.0, np.inf)
        
        # Engle-Granger cointegration test
        _, p_value, _ = coint(df['s1'], df['s2'])
        
        # Calculate hedge ratio using linear regression
        model = OLS(df['s1'], df['s2']).fit()
        hedge_ratio = model.params[0]
        
        # Calculate spread
        spread = df['s1'] - hedge_ratio * df['s2']
        
        # Calculate half-life of mean reversion
        half_life = self._calculate_half_life(spread)
        
        return (p_value, hedge_ratio, half_life)
    
    def _calculate_half_life(self, spread: pd.Series) -> float:
        """
        Calculate half-life of mean reversion.
        
        Args:
            spread: Spread time series
            
        Returns:
            Half-life in days
        """
        try:
            spread_lag = spread.shift(1).dropna()
            spread_delta = spread.diff().dropna()
            
            # Align series
            spread_lag = spread_lag[spread_delta.index]
            
            # Regression: delta(spread) = lambda * spread(t-1) + epsilon
            model = OLS(spread_delta, spread_lag).fit()
            lambda_param = model.params[0]
            
            if lambda_param >= 0:
                return np.inf
            
            half_life = -np.log(2) / lambda_param
            return half_life
            
        except:
            return np.inf
    
    def find_pairs(
        self,
        price_data: pd.DataFrame,
        tickers: List[str] = None
    ) -> List[Dict]:
        """
        Find cointegrated pairs from price data.
        
        Args:
            price_data: DataFrame with columns ['ticker', 'date', 'adj_close']
            tickers: List of tickers to analyze (if None, use all)
            
        Returns:
            List of pair dictionaries
        """
        # Pivot data to wide format
        prices = price_data.pivot(
            index='date',
            columns='ticker',
            values='adj_close'
        )
        
        if tickers is None:
            tickers = prices.columns.tolist()
        
        # Filter to recent data
        if len(prices) > self.lookback_period:
            prices = prices.tail(self.lookback_period)
        
        # Test all combinations
        pairs = []
        ticker_combinations = list(combinations(tickers, 2))
        
        print(f"Testing {len(ticker_combinations)} ticker combinations...")
        
        for ticker1, ticker2 in tqdm(ticker_combinations):
            if ticker1 not in prices.columns or ticker2 not in prices.columns:
                continue
            
            series1 = prices[ticker1].dropna()
            series2 = prices[ticker2].dropna()
            
            # Skip if insufficient data
            if len(series1) < 30 or len(series2) < 30:
                continue
            
            # Test cointegration
            p_value, hedge_ratio, half_life = self.test_cointegration(series1, series2)
            
            # Check if pair qualifies
            if p_value < self.p_value_threshold and 1 <= half_life <= 60:
                # Calculate correlation
                correlation = series1.corr(series2)
                
                pairs.append({
                    'ticker_1': ticker1,
                    'ticker_2': ticker2,
                    'p_value': p_value,
                    'hedge_ratio': hedge_ratio,
                    'half_life': half_life,
                    'correlation': correlation,
                    'method': 'cointegration'
                })
        
        # Sort by p-value (best pairs first)
        pairs.sort(key=lambda x: x['p_value'])
        
        print(f"✓ Found {len(pairs)} cointegrated pairs")
        return pairs


class ClusteringPairFinder:
    """Find pairs using clustering algorithms."""
    
    def __init__(
        self,
        method: str = 'kmeans',
        n_clusters: int = 20,
        lookback_period: int = 252
    ):
        """
        Initialize clustering pair finder.
        
        Args:
            method: 'kmeans' or 'dbscan'
            n_clusters: Number of clusters (for k-means)
            lookback_period: Number of trading days
        """
        self.method = method
        self.n_clusters = n_clusters
        self.lookback_period = lookback_period
    
    def calculate_features(
        self,
        price_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate features for clustering.
        
        Args:
            price_data: DataFrame with price data
            
        Returns:
            DataFrame with features for each ticker
        """
        # Pivot to wide format
        prices = price_data.pivot(
            index='date',
            columns='ticker',
            values='adj_close'
        )
        
        # Filter to recent data
        if len(prices) > self.lookback_period:
            prices = prices.tail(self.lookback_period)
        
        # Calculate returns
        returns = prices.pct_change().dropna()
        
        features = []
        
        for ticker in prices.columns:
            ticker_returns = returns[ticker].dropna()
            
            if len(ticker_returns) < 30:
                continue
            
            features.append({
                'ticker': ticker,
                'mean_return': ticker_returns.mean(),
                'volatility': ticker_returns.std(),
                'skewness': ticker_returns.skew(),
                'kurtosis': ticker_returns.kurt()
            })
        
        return pd.DataFrame(features)
    
    def find_pairs(
        self,
        price_data: pd.DataFrame
    ) -> List[Dict]:
        """
        Find pairs using clustering.
        
        Args:
            price_data: DataFrame with price data
            
        Returns:
            List of pair dictionaries
        """
        # Calculate features
        features_df = self.calculate_features(price_data)
        
        if len(features_df) < 10:
            print("⚠ Insufficient data for clustering")
            return []
        
        # Prepare features for clustering
        feature_cols = ['mean_return', 'volatility', 'skewness', 'kurtosis']
        X = features_df[feature_cols].values
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform clustering
        if self.method == 'kmeans':
            clusterer = KMeans(n_clusters=self.n_clusters, random_state=42)
        elif self.method == 'dbscan':
            clusterer = DBSCAN(eps=0.5, min_samples=2)
        else:
            raise ValueError(f"Unknown clustering method: {self.method}")
        
        labels = clusterer.fit_predict(X_scaled)
        features_df['cluster'] = labels
        
        # Find pairs within each cluster
        pairs = []
        
        for cluster_id in features_df['cluster'].unique():
            if cluster_id == -1:  # Skip noise points (DBSCAN)
                continue
            
            cluster_tickers = features_df[features_df['cluster'] == cluster_id]['ticker'].tolist()
            
            if len(cluster_tickers) < 2:
                continue
            
            # Create pairs from tickers in same cluster
            for ticker1, ticker2 in combinations(cluster_tickers, 2):
                pairs.append({
                    'ticker_1': ticker1,
                    'ticker_2': ticker2,
                    'cluster_id': int(cluster_id),
                    'method': 'clustering'
                })
        
        print(f"✓ Found {len(pairs)} pairs from {len(features_df['cluster'].unique())} clusters")
        return pairs


def calculate_spread(
    price_data: pd.DataFrame,
    ticker1: str,
    ticker2: str,
    hedge_ratio: float = 1.0
) -> pd.DataFrame:
    """
    Calculate spread between two stocks.
    
    Args:
        price_data: DataFrame with price data
        ticker1: First ticker
        ticker2: Second ticker
        hedge_ratio: Hedge ratio from cointegration test
        
    Returns:
        DataFrame with spread and z-score
    """
    # Get prices
    prices = price_data.pivot(
        index='date',
        columns='ticker',
        values='adj_close'
    )
    
    # Calculate spread
    spread = prices[ticker1] - hedge_ratio * prices[ticker2]
    
    # Calculate rolling statistics
    window = 60
    spread_mean = spread.rolling(window=window).mean()
    spread_std = spread.rolling(window=window).std()
    
    # Calculate z-score
    zscore = (spread - spread_mean) / spread_std
    
    # Create DataFrame
    result = pd.DataFrame({
        'date': spread.index,
        'spread': spread.values,
        'mean': spread_mean.values,
        'std_dev': spread_std.values,
        'zscore': zscore.values
    })
    
    return result.dropna()


if __name__ == "__main__":
    # Example usage
    print("Pair Identification Module")
    print("-" * 50)
    
    # This would normally use real data
    # Here's just a demonstration of the API
    
    finder = CointegrationPairFinder(p_value_threshold=0.05)
    print(f"Initialized CointegrationPairFinder")
    
    cluster_finder = ClusteringPairFinder(method='kmeans', n_clusters=20)
    print(f"Initialized ClusteringPairFinder")
