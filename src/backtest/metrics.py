"""
Performance metrics and analysis for backtesting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns


class PerformanceAnalyzer:
    """Analyze backtest performance and calculate metrics."""
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize performance analyzer.
        
        Args:
            risk_free_rate: Annual risk-free rate for Sharpe ratio
        """
        self.risk_free_rate = risk_free_rate
    
    def calculate_returns(self, performance_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate daily and cumulative returns.
        
        Args:
            performance_df: DataFrame with portfolio values
            
        Returns:
            DataFrame with returns added
        """
        df = performance_df.copy()
        
        # Daily returns
        df['daily_return'] = df['portfolio_value'].pct_change()
        
        # Cumulative returns
        df['cumulative_return'] = (1 + df['daily_return']).cumprod() - 1
        
        return df
    
    def calculate_sharpe_ratio(
        self,
        returns: pd.Series,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            returns: Series of returns
            periods_per_year: Number of periods per year (252 for daily)
            
        Returns:
            Sharpe ratio
        """
        excess_returns = returns - (self.risk_free_rate / periods_per_year)
        
        if excess_returns.std() == 0:
            return 0.0
        
        sharpe = np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()
        
        return sharpe
    
    def calculate_max_drawdown(self, cumulative_returns: pd.Series) -> float:
        """
        Calculate maximum drawdown.
        
        Args:
            cumulative_returns: Series of cumulative returns
            
        Returns:
            Maximum drawdown (positive number)
        """
        # Convert to wealth index
        wealth_index = 1 + cumulative_returns
        
        # Calculate running maximum
        running_max = wealth_index.expanding().max()
        
        # Calculate drawdown
        drawdown = (wealth_index - running_max) / running_max
        
        return abs(drawdown.min())
    
    def calculate_sortino_ratio(
        self,
        returns: pd.Series,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Sortino ratio (like Sharpe but only penalizes downside volatility).
        
        Args:
            returns: Series of returns
            periods_per_year: Number of periods per year
            
        Returns:
            Sortino ratio
        """
        excess_returns = returns - (self.risk_free_rate / periods_per_year)
        
        # Downside deviation (only negative returns)
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0 or downside_returns.std() == 0:
            return 0.0
        
        sortino = np.sqrt(periods_per_year) * excess_returns.mean() / downside_returns.std()
        
        return sortino
    
    def calculate_win_rate(self, trades: List) -> float:
        """
        Calculate win rate.
        
        Args:
            trades: List of Trade objects
            
        Returns:
            Win rate as decimal
        """
        if len(trades) == 0:
            return 0.0
        
        winning_trades = sum(1 for trade in trades if trade.pnl > 0)
        
        return winning_trades / len(trades)
    
    def calculate_profit_factor(self, trades: List) -> float:
        """
        Calculate profit factor (gross profit / gross loss).
        
        Args:
            trades: List of Trade objects
            
        Returns:
            Profit factor
        """
        gross_profit = sum(trade.pnl for trade in trades if trade.pnl > 0)
        gross_loss = abs(sum(trade.pnl for trade in trades if trade.pnl < 0))
        
        if gross_loss == 0:
            return np.inf if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    def calculate_all_metrics(
        self,
        performance_df: pd.DataFrame,
        trades: List
    ) -> Dict[str, float]:
        """
        Calculate all performance metrics.
        
        Args:
            performance_df: DataFrame with portfolio performance
            trades: List of Trade objects
            
        Returns:
            Dictionary of metrics
        """
        # Calculate returns
        df = self.calculate_returns(performance_df)
        
        # Filter out NaN returns
        returns = df['daily_return'].dropna()
        
        # Calculate metrics
        metrics = {
            'total_return': df['cumulative_return'].iloc[-1] if len(df) > 0 else 0,
            'annual_return': (1 + df['cumulative_return'].iloc[-1]) ** (252 / len(df)) - 1 if len(df) > 0 else 0,
            'sharpe_ratio': self.calculate_sharpe_ratio(returns),
            'sortino_ratio': self.calculate_sortino_ratio(returns),
            'max_drawdown': self.calculate_max_drawdown(df['cumulative_return']),
            'volatility': returns.std() * np.sqrt(252),
            'win_rate': self.calculate_win_rate(trades),
            'profit_factor': self.calculate_profit_factor(trades),
            'num_trades': len(trades),
            'avg_trade_return': np.mean([trade.pnl_percent for trade in trades]) if trades else 0,
            'final_portfolio_value': df['portfolio_value'].iloc[-1] if len(df) > 0 else 0
        }
        
        return metrics
    
    def print_metrics(self, metrics: Dict[str, float]):
        """Print metrics in a formatted way."""
        print("\n" + "="*60)
        print("PERFORMANCE METRICS")
        print("="*60)
        
        print(f"\nReturns:")
        print(f"  Total Return:        {metrics['total_return']*100:>8.2f}%")
        print(f"  Annual Return:       {metrics['annual_return']*100:>8.2f}%")
        
        print(f"\nRisk-Adjusted:")
        print(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:>8.2f}")
        print(f"  Sortino Ratio:       {metrics['sortino_ratio']:>8.2f}")
        print(f"  Max Drawdown:        {metrics['max_drawdown']*100:>8.2f}%")
        print(f"  Volatility (annual): {metrics['volatility']*100:>8.2f}%")
        
        print(f"\nTrading:")
        print(f"  Number of Trades:    {metrics['num_trades']:>8.0f}")
        print(f"  Win Rate:            {metrics['win_rate']*100:>8.2f}%")
        print(f"  Profit Factor:       {metrics['profit_factor']:>8.2f}")
        print(f"  Avg Trade Return:    {metrics['avg_trade_return']*100:>8.2f}%")
        
        print(f"\nPortfolio:")
        print(f"  Final Value:         ${metrics['final_portfolio_value']:>12,.2f}")
        
        print("="*60 + "\n")
    
    def plot_performance(self, performance_df: pd.DataFrame, save_path: str = None):
        """
        Plot performance charts.
        
        Args:
            performance_df: DataFrame with performance data
            save_path: Path to save plot (optional)
        """
        df = self.calculate_returns(performance_df)
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Portfolio value
        axes[0].plot(df['date'], df['portfolio_value'], linewidth=2)
        axes[0].set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Portfolio Value ($)')
        axes[0].grid(True, alpha=0.3)
        
        # Cumulative returns
        axes[1].plot(df['date'], df['cumulative_return'] * 100, linewidth=2, color='green')
        axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.3)
        axes[1].set_title('Cumulative Returns', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Return (%)')
        axes[1].grid(True, alpha=0.3)
        
        # Drawdown
        wealth_index = 1 + df['cumulative_return']
        running_max = wealth_index.expanding().max()
        drawdown = (wealth_index - running_max) / running_max
        
        axes[2].fill_between(df['date'], drawdown * 100, 0, color='red', alpha=0.3)
        axes[2].set_title('Drawdown', fontsize=14, fontweight='bold')
        axes[2].set_ylabel('Drawdown (%)')
        axes[2].set_xlabel('Date')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_trade_distribution(self, trades: List, save_path: str = None):
        """
        Plot distribution of trade returns.
        
        Args:
            trades: List of Trade objects
            save_path: Path to save plot
        """
        if not trades:
            print("No trades to plot")
            return
        
        returns = [trade.pnl_percent * 100 for trade in trades]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(returns, bins=30, edgecolor='black', alpha=0.7)
        axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[0].set_title('Distribution of Trade Returns', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Return (%)')
        axes[0].set_ylabel('Frequency')
        axes[0].grid(True, alpha=0.3)
        
        # Box plot
        axes[1].boxplot(returns, vert=True)
        axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2)
        axes[1].set_title('Trade Returns Box Plot', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Return (%)')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved to {save_path}")
        else:
            plt.show()


# Standalone helper functions for easy imports
def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02, periods_per_year: int = 252) -> float:
    """
    Calculate Sharpe ratio from returns array.
    
    Args:
        returns: Array of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year (252 for daily)
        
    Returns:
        Sharpe ratio
    """
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - (risk_free_rate / periods_per_year)
    
    if np.std(excess_returns) == 0:
        return 0.0
    
    sharpe = np.sqrt(periods_per_year) * np.mean(excess_returns) / np.std(excess_returns)
    
    return sharpe


def calculate_max_drawdown(cumulative_returns: np.ndarray) -> float:
    """
    Calculate maximum drawdown from cumulative returns.
    
    Args:
        cumulative_returns: Array of cumulative returns
        
    Returns:
        Maximum drawdown as percentage (positive number)
    """
    if len(cumulative_returns) == 0:
        return 0.0
    
    # Convert to wealth index
    wealth_index = 1 + cumulative_returns
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(wealth_index)
    
    # Calculate drawdown
    drawdown = (wealth_index - running_max) / running_max
    
    return abs(np.min(drawdown)) * 100  # Return as percentage


def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.02, periods_per_year: int = 252) -> float:
    """
    Calculate Sortino ratio from returns array.
    
    Args:
        returns: Array of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year
        
    Returns:
        Sortino ratio
    """
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - (risk_free_rate / periods_per_year)
    
    # Downside deviation (only negative returns)
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0 or np.std(downside_returns) == 0:
        return 0.0
    
    sortino = np.sqrt(periods_per_year) * np.mean(excess_returns) / np.std(downside_returns)
    
    return sortino


if __name__ == "__main__":
    print("Performance Metrics Module")
    print("-" * 50)
    
    analyzer = PerformanceAnalyzer(risk_free_rate=0.02)
    print("✓ Initialized PerformanceAnalyzer")
