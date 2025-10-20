-- Database schema for pairs trading system

-- Stock price data
CREATE TABLE IF NOT EXISTS stock_prices (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    open DECIMAL(12, 4),
    high DECIMAL(12, 4),
    low DECIMAL(12, 4),
    close DECIMAL(12, 4),
    adj_close DECIMAL(12, 4),
    volume BIGINT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ticker, date)
);

CREATE INDEX idx_stock_prices_ticker_date ON stock_prices(ticker, date DESC);

-- News articles
CREATE TABLE IF NOT EXISTS news_articles (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    content TEXT,
    url TEXT,
    published_at TIMESTAMP NOT NULL,
    source VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_news_ticker_date ON news_articles(ticker, published_at DESC);

-- Sentiment scores
CREATE TABLE IF NOT EXISTS sentiment_scores (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    sentiment_score DECIMAL(5, 4),  -- -1 to 1
    compound_score DECIMAL(5, 4),
    positive_score DECIMAL(5, 4),
    negative_score DECIMAL(5, 4),
    neutral_score DECIMAL(5, 4),
    article_count INT DEFAULT 0,
    model_version VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ticker, date)
);

CREATE INDEX idx_sentiment_ticker_date ON sentiment_scores(ticker, date DESC);

-- Identified pairs
CREATE TABLE IF NOT EXISTS stock_pairs (
    id SERIAL PRIMARY KEY,
    ticker_1 VARCHAR(10) NOT NULL,
    ticker_2 VARCHAR(10) NOT NULL,
    method VARCHAR(50),  -- 'cointegration' or 'clustering'
    p_value DECIMAL(6, 4),
    correlation DECIMAL(5, 4),
    half_life DECIMAL(8, 2),
    hedge_ratio DECIMAL(10, 6),
    identified_date DATE NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ticker_1, ticker_2, identified_date)
);

CREATE INDEX idx_pairs_active ON stock_pairs(is_active, identified_date DESC);

-- Spread data
CREATE TABLE IF NOT EXISTS pair_spreads (
    id SERIAL PRIMARY KEY,
    pair_id INT REFERENCES stock_pairs(id),
    date DATE NOT NULL,
    spread DECIMAL(12, 6),
    zscore DECIMAL(8, 4),
    mean DECIMAL(12, 6),
    std_dev DECIMAL(12, 6),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(pair_id, date)
);

CREATE INDEX idx_spreads_pair_date ON pair_spreads(pair_id, date DESC);

-- Trading signals
CREATE TABLE IF NOT EXISTS trading_signals (
    id SERIAL PRIMARY KEY,
    pair_id INT REFERENCES stock_pairs(id),
    date DATE NOT NULL,
    signal VARCHAR(10),  -- 'LONG', 'SHORT', 'EXIT', 'HOLD'
    confidence DECIMAL(5, 4),
    predicted_spread DECIMAL(12, 6),
    features JSONB,
    model_version VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_signals_pair_date ON trading_signals(pair_id, date DESC);

-- Trades (executed or simulated)
CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    pair_id INT REFERENCES stock_pairs(id),
    signal_id INT REFERENCES trading_signals(id),
    entry_date DATE NOT NULL,
    exit_date DATE,
    ticker_1_entry_price DECIMAL(12, 4),
    ticker_2_entry_price DECIMAL(12, 4),
    ticker_1_exit_price DECIMAL(12, 4),
    ticker_2_exit_price DECIMAL(12, 4),
    position_size DECIMAL(12, 2),
    pnl DECIMAL(12, 2),
    pnl_percent DECIMAL(8, 4),
    status VARCHAR(20),  -- 'OPEN', 'CLOSED', 'STOPPED'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_trades_status_date ON trades(status, entry_date DESC);

-- Portfolio performance
CREATE TABLE IF NOT EXISTS portfolio_performance (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL UNIQUE,
    portfolio_value DECIMAL(15, 2),
    cash_balance DECIMAL(15, 2),
    daily_return DECIMAL(8, 6),
    cumulative_return DECIMAL(8, 6),
    sharpe_ratio DECIMAL(8, 4),
    max_drawdown DECIMAL(8, 4),
    num_positions INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_performance_date ON portfolio_performance(date DESC);
