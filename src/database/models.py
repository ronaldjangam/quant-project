"""
SQLAlchemy ORM models for the database.
"""

from datetime import datetime, date
from sqlalchemy import Column, Integer, String, Date, DateTime, Numeric, BigInteger, Boolean, Text, ForeignKey, JSON
from sqlalchemy.orm import relationship
from .connection import Base


class StockPrice(Base):
    """Stock price data model."""
    __tablename__ = 'stock_prices'
    
    id = Column(Integer, primary_key=True)
    ticker = Column(String(10), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    open = Column(Numeric(12, 4))
    high = Column(Numeric(12, 4))
    low = Column(Numeric(12, 4))
    close = Column(Numeric(12, 4))
    adj_close = Column(Numeric(12, 4))
    volume = Column(BigInteger)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<StockPrice(ticker={self.ticker}, date={self.date}, close={self.close})>"


class NewsArticle(Base):
    """News article model."""
    __tablename__ = 'news_articles'
    
    id = Column(Integer, primary_key=True)
    ticker = Column(String(10), nullable=False, index=True)
    title = Column(Text, nullable=False)
    description = Column(Text)
    content = Column(Text)
    url = Column(Text)
    published_at = Column(DateTime, nullable=False, index=True)
    source = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<NewsArticle(ticker={self.ticker}, title={self.title[:50]})>"


class SentimentScore(Base):
    """Sentiment score model."""
    __tablename__ = 'sentiment_scores'
    
    id = Column(Integer, primary_key=True)
    ticker = Column(String(10), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    sentiment_score = Column(Numeric(5, 4))
    compound_score = Column(Numeric(5, 4))
    positive_score = Column(Numeric(5, 4))
    negative_score = Column(Numeric(5, 4))
    neutral_score = Column(Numeric(5, 4))
    article_count = Column(Integer, default=0)
    model_version = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<SentimentScore(ticker={self.ticker}, date={self.date}, score={self.sentiment_score})>"


class StockPair(Base):
    """Stock pair model."""
    __tablename__ = 'stock_pairs'
    
    id = Column(Integer, primary_key=True)
    ticker_1 = Column(String(10), nullable=False)
    ticker_2 = Column(String(10), nullable=False)
    method = Column(String(50))
    p_value = Column(Numeric(6, 4))
    correlation = Column(Numeric(5, 4))
    half_life = Column(Numeric(8, 2))
    hedge_ratio = Column(Numeric(10, 6))
    identified_date = Column(Date, nullable=False)
    is_active = Column(Boolean, default=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    spreads = relationship("PairSpread", back_populates="pair")
    signals = relationship("TradingSignal", back_populates="pair")
    trades = relationship("Trade", back_populates="pair")
    
    def __repr__(self):
        return f"<StockPair({self.ticker_1}/{self.ticker_2}, p={self.p_value})>"


class PairSpread(Base):
    """Pair spread data model."""
    __tablename__ = 'pair_spreads'
    
    id = Column(Integer, primary_key=True)
    pair_id = Column(Integer, ForeignKey('stock_pairs.id'), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    spread = Column(Numeric(12, 6))
    zscore = Column(Numeric(8, 4))
    mean = Column(Numeric(12, 6))
    std_dev = Column(Numeric(12, 6))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    pair = relationship("StockPair", back_populates="spreads")
    
    def __repr__(self):
        return f"<PairSpread(pair_id={self.pair_id}, date={self.date}, zscore={self.zscore})>"


class TradingSignal(Base):
    """Trading signal model."""
    __tablename__ = 'trading_signals'
    
    id = Column(Integer, primary_key=True)
    pair_id = Column(Integer, ForeignKey('stock_pairs.id'), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    signal = Column(String(10))  # LONG, SHORT, EXIT, HOLD
    confidence = Column(Numeric(5, 4))
    predicted_spread = Column(Numeric(12, 6))
    features = Column(JSON)
    model_version = Column(String(50))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    pair = relationship("StockPair", back_populates="signals")
    trades = relationship("Trade", back_populates="signal")
    
    def __repr__(self):
        return f"<TradingSignal(pair_id={self.pair_id}, signal={self.signal}, confidence={self.confidence})>"


class Trade(Base):
    """Trade model."""
    __tablename__ = 'trades'
    
    id = Column(Integer, primary_key=True)
    pair_id = Column(Integer, ForeignKey('stock_pairs.id'), nullable=False)
    signal_id = Column(Integer, ForeignKey('trading_signals.id'))
    entry_date = Column(Date, nullable=False, index=True)
    exit_date = Column(Date)
    ticker_1_entry_price = Column(Numeric(12, 4))
    ticker_2_entry_price = Column(Numeric(12, 4))
    ticker_1_exit_price = Column(Numeric(12, 4))
    ticker_2_exit_price = Column(Numeric(12, 4))
    position_size = Column(Numeric(12, 2))
    pnl = Column(Numeric(12, 2))
    pnl_percent = Column(Numeric(8, 4))
    status = Column(String(20), index=True)  # OPEN, CLOSED, STOPPED
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    pair = relationship("StockPair", back_populates="trades")
    signal = relationship("TradingSignal", back_populates="trades")
    
    def __repr__(self):
        return f"<Trade(pair_id={self.pair_id}, status={self.status}, pnl={self.pnl})>"


class PortfolioPerformance(Base):
    """Portfolio performance model."""
    __tablename__ = 'portfolio_performance'
    
    id = Column(Integer, primary_key=True)
    date = Column(Date, nullable=False, unique=True, index=True)
    portfolio_value = Column(Numeric(15, 2))
    cash_balance = Column(Numeric(15, 2))
    daily_return = Column(Numeric(8, 6))
    cumulative_return = Column(Numeric(8, 6))
    sharpe_ratio = Column(Numeric(8, 4))
    max_drawdown = Column(Numeric(8, 4))
    num_positions = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<PortfolioPerformance(date={self.date}, value={self.portfolio_value})>"
