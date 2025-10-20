"""
Event-driven backtesting engine for pairs trading.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum


class SignalType(Enum):
    """Trading signal types."""
    LONG = "LONG"
    SHORT = "SHORT"
    EXIT = "EXIT"
    HOLD = "HOLD"


@dataclass
class Position:
    """Represents an open position."""
    pair_id: int
    ticker_1: str
    ticker_2: str
    entry_date: datetime
    entry_price_1: float
    entry_price_2: float
    position_size: float
    signal_type: SignalType
    hedge_ratio: float
    entry_zscore: float
    stop_loss_zscore: float = 3.5


@dataclass
class Trade:
    """Represents a closed trade."""
    pair_id: int
    ticker_1: str
    ticker_2: str
    entry_date: datetime
    exit_date: datetime
    entry_price_1: float
    entry_price_2: float
    exit_price_1: float
    exit_price_2: float
    position_size: float
    pnl: float
    pnl_percent: float
    signal_type: SignalType
    exit_reason: str


class Portfolio:
    """Manages portfolio state and calculations."""
    
    def __init__(
        self,
        initial_capital: float = 100000,
        max_positions: int = 10,
        max_position_size: float = 0.2
    ):
        """
        Initialize portfolio.
        
        Args:
            initial_capital: Starting capital
            max_positions: Maximum concurrent positions
            max_position_size: Maximum % of portfolio per position
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.max_positions = max_positions
        self.max_position_size = max_position_size
        
        self.positions: Dict[int, Position] = {}
        self.closed_trades: List[Trade] = []
        
        self.daily_values = []
        self.daily_returns = []
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate current portfolio value.
        
        Args:
            current_prices: Dict mapping ticker to current price
            
        Returns:
            Total portfolio value
        """
        value = self.cash
        
        for position in self.positions.values():
            price_1 = current_prices.get(position.ticker_1, position.entry_price_1)
            price_2 = current_prices.get(position.ticker_2, position.entry_price_2)
            
            # Calculate position value
            if position.signal_type == SignalType.LONG:
                # Long stock 1, short stock 2
                pnl = position.position_size * (
                    (price_1 - position.entry_price_1) - 
                    position.hedge_ratio * (price_2 - position.entry_price_2)
                )
            else:  # SHORT
                # Short stock 1, long stock 2
                pnl = position.position_size * (
                    (position.entry_price_1 - price_1) - 
                    position.hedge_ratio * (position.entry_price_2 - price_2)
                )
            
            value += pnl
        
        return value
    
    def calculate_position_size(self) -> float:
        """
        Calculate position size based on available capital.
        
        Returns:
            Position size in dollars
        """
        portfolio_value = self.cash  # Simplified
        max_size = portfolio_value * self.max_position_size
        
        # Divide by number of positions we can have
        if len(self.positions) < self.max_positions:
            return max_size
        else:
            return 0
    
    def can_open_position(self) -> bool:
        """Check if we can open a new position."""
        return len(self.positions) < self.max_positions and self.cash > 0


class BacktestEngine:
    """Event-driven backtesting engine."""
    
    def __init__(
        self,
        initial_capital: float = 100000,
        transaction_cost: float = 0.001,
        slippage: float = 0.0005,
        entry_zscore_threshold: float = 2.0,
        exit_zscore_threshold: float = 0.5,
        stop_loss_zscore: float = 3.5,
        max_positions: int = 10
    ):
        """
        Initialize backtest engine.
        
        Args:
            initial_capital: Starting capital
            transaction_cost: Transaction cost as fraction (0.001 = 0.1%)
            slippage: Slippage as fraction
            entry_zscore_threshold: Z-score threshold for entry
            exit_zscore_threshold: Z-score threshold for exit
            stop_loss_zscore: Stop loss z-score
            max_positions: Maximum concurrent positions
        """
        self.portfolio = Portfolio(
            initial_capital=initial_capital,
            max_positions=max_positions
        )
        
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.entry_zscore_threshold = entry_zscore_threshold
        self.exit_zscore_threshold = exit_zscore_threshold
        self.stop_loss_zscore = stop_loss_zscore
        
        self.performance_history = []
    
    def apply_costs(self, price: float, is_buy: bool = True) -> float:
        """
        Apply transaction costs and slippage.
        
        Args:
            price: Original price
            is_buy: Whether this is a buy (True) or sell (False)
            
        Returns:
            Adjusted price
        """
        # Slippage: pay more when buying, receive less when selling
        if is_buy:
            price *= (1 + self.slippage)
        else:
            price *= (1 - self.slippage)
        
        # Transaction cost
        price *= (1 + self.transaction_cost)
        
        return price
    
    def generate_signal(
        self,
        zscore: float,
        predicted_zscore: float = None
    ) -> SignalType:
        """
        Generate trading signal based on z-score.
        
        Args:
            zscore: Current z-score
            predicted_zscore: Predicted z-score (if using ML)
            
        Returns:
            Trading signal
        """
        # Use ML prediction if available
        if predicted_zscore is not None:
            # If model predicts mean reversion
            if zscore < -self.entry_zscore_threshold and predicted_zscore > zscore:
                return SignalType.LONG
            elif zscore > self.entry_zscore_threshold and predicted_zscore < zscore:
                return SignalType.SHORT
        else:
            # Rule-based signals
            if zscore < -self.entry_zscore_threshold:
                return SignalType.LONG
            elif zscore > self.entry_zscore_threshold:
                return SignalType.SHORT
        
        # Exit signals
        if abs(zscore) < self.exit_zscore_threshold:
            return SignalType.EXIT
        
        return SignalType.HOLD
    
    def open_position(
        self,
        date: datetime,
        pair_id: int,
        ticker_1: str,
        ticker_2: str,
        price_1: float,
        price_2: float,
        hedge_ratio: float,
        zscore: float,
        signal: SignalType
    ):
        """Open a new position."""
        if not self.portfolio.can_open_position():
            return
        
        position_size = self.portfolio.calculate_position_size()
        
        if position_size <= 0:
            return
        
        # Apply costs
        entry_price_1 = self.apply_costs(price_1, is_buy=(signal == SignalType.LONG))
        entry_price_2 = self.apply_costs(price_2, is_buy=(signal == SignalType.SHORT))
        
        # Create position
        position = Position(
            pair_id=pair_id,
            ticker_1=ticker_1,
            ticker_2=ticker_2,
            entry_date=date,
            entry_price_1=entry_price_1,
            entry_price_2=entry_price_2,
            position_size=position_size,
            signal_type=signal,
            hedge_ratio=hedge_ratio,
            entry_zscore=zscore,
            stop_loss_zscore=self.stop_loss_zscore
        )
        
        # Update portfolio
        self.portfolio.positions[pair_id] = position
        self.portfolio.cash -= position_size
    
    def close_position(
        self,
        date: datetime,
        pair_id: int,
        price_1: float,
        price_2: float,
        exit_reason: str = "signal"
    ):
        """Close an existing position."""
        if pair_id not in self.portfolio.positions:
            return
        
        position = self.portfolio.positions[pair_id]
        
        # Apply costs
        exit_price_1 = self.apply_costs(
            price_1,
            is_buy=(position.signal_type == SignalType.SHORT)
        )
        exit_price_2 = self.apply_costs(
            price_2,
            is_buy=(position.signal_type == SignalType.LONG)
        )
        
        # Calculate P&L
        if position.signal_type == SignalType.LONG:
            pnl = position.position_size * (
                (exit_price_1 - position.entry_price_1) - 
                position.hedge_ratio * (exit_price_2 - position.entry_price_2)
            )
        else:  # SHORT
            pnl = position.position_size * (
                (position.entry_price_1 - exit_price_1) - 
                position.hedge_ratio * (position.entry_price_2 - exit_price_2)
            )
        
        pnl_percent = pnl / position.position_size
        
        # Create trade record
        trade = Trade(
            pair_id=pair_id,
            ticker_1=position.ticker_1,
            ticker_2=position.ticker_2,
            entry_date=position.entry_date,
            exit_date=date,
            entry_price_1=position.entry_price_1,
            entry_price_2=position.entry_price_2,
            exit_price_1=exit_price_1,
            exit_price_2=exit_price_2,
            position_size=position.position_size,
            pnl=pnl,
            pnl_percent=pnl_percent,
            signal_type=position.signal_type,
            exit_reason=exit_reason
        )
        
        # Update portfolio
        self.portfolio.closed_trades.append(trade)
        self.portfolio.cash += position.position_size + pnl
        del self.portfolio.positions[pair_id]
    
    def check_stop_loss(
        self,
        date: datetime,
        pair_id: int,
        current_zscore: float,
        price_1: float,
        price_2: float
    ):
        """Check if position should be stopped out."""
        if pair_id not in self.portfolio.positions:
            return
        
        position = self.portfolio.positions[pair_id]
        
        # Check if z-score has moved against us
        if position.signal_type == SignalType.LONG:
            if current_zscore < -position.stop_loss_zscore:
                self.close_position(date, pair_id, price_1, price_2, "stop_loss")
        else:  # SHORT
            if current_zscore > position.stop_loss_zscore:
                self.close_position(date, pair_id, price_1, price_2, "stop_loss")
    
    def run_backtest(
        self,
        pairs_data: pd.DataFrame,
        price_data: pd.DataFrame,
        signals_data: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Run backtest on historical data.
        
        Args:
            pairs_data: DataFrame with pair information
            price_data: DataFrame with price data
            signals_data: DataFrame with ML predictions (optional)
            
        Returns:
            DataFrame with performance metrics
        """
        print("Running backtest...")
        
        # Get all dates
        dates = sorted(price_data['date'].unique())
        
        for current_date in dates:
            # Get current prices
            day_prices = price_data[price_data['date'] == current_date]
            current_prices = dict(zip(day_prices['ticker'], day_prices['adj_close']))
            
            # Update portfolio value
            portfolio_value = self.portfolio.get_portfolio_value(current_prices)
            
            # Record performance
            self.performance_history.append({
                'date': current_date,
                'portfolio_value': portfolio_value,
                'cash': self.portfolio.cash,
                'num_positions': len(self.portfolio.positions)
            })
            
            # Check each pair
            for _, pair_row in pairs_data.iterrows():
                pair_id = pair_row['id']
                ticker_1 = pair_row['ticker_1']
                ticker_2 = pair_row['ticker_2']
                hedge_ratio = pair_row.get('hedge_ratio', 1.0)
                
                # Get current prices
                if ticker_1 not in current_prices or ticker_2 not in current_prices:
                    continue
                
                price_1 = current_prices[ticker_1]
                price_2 = current_prices[ticker_2]
                
                # Get current z-score (simplified - should come from spread data)
                spread = price_1 - hedge_ratio * price_2
                zscore = 0  # Placeholder
                
                # Check stop loss for existing positions
                if pair_id in self.portfolio.positions:
                    self.check_stop_loss(current_date, pair_id, zscore, price_1, price_2)
                
                # Generate signal
                predicted_zscore = None
                if signals_data is not None:
                    signal_row = signals_data[
                        (signals_data['pair_id'] == pair_id) & 
                        (signals_data['date'] == current_date)
                    ]
                    if not signal_row.empty:
                        predicted_zscore = signal_row.iloc[0]['predicted_zscore']
                
                signal = self.generate_signal(zscore, predicted_zscore)
                
                # Execute trades
                if pair_id not in self.portfolio.positions:
                    if signal in [SignalType.LONG, SignalType.SHORT]:
                        self.open_position(
                            current_date, pair_id, ticker_1, ticker_2,
                            price_1, price_2, hedge_ratio, zscore, signal
                        )
                else:
                    if signal == SignalType.EXIT:
                        self.close_position(current_date, pair_id, price_1, price_2)
        
        # Close all remaining positions at the end
        final_date = dates[-1]
        final_prices = price_data[price_data['date'] == final_date]
        final_prices_dict = dict(zip(final_prices['ticker'], final_prices['adj_close']))
        
        for pair_id in list(self.portfolio.positions.keys()):
            position = self.portfolio.positions[pair_id]
            self.close_position(
                final_date,
                pair_id,
                final_prices_dict.get(position.ticker_1, position.entry_price_1),
                final_prices_dict.get(position.ticker_2, position.entry_price_2),
                "end_of_backtest"
            )
        
        # Convert to DataFrame
        performance_df = pd.DataFrame(self.performance_history)
        
        print(f"✓ Backtest complete. {len(self.portfolio.closed_trades)} trades executed.")
        
        return performance_df


if __name__ == "__main__":
    print("Backtesting Engine Module")
    print("-" * 50)
    
    engine = BacktestEngine(
        initial_capital=100000,
        transaction_cost=0.001,
        slippage=0.0005
    )
    
    print(f"✓ Initialized BacktestEngine")
    print(f"  Initial capital: ${engine.portfolio.initial_capital:,.2f}")
    print(f"  Transaction cost: {engine.transaction_cost*100:.2f}%")
    print(f"  Slippage: {engine.slippage*100:.3f}%")
