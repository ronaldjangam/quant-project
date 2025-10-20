# AI-Powered Pairs Trading Strategy 🚀

An advanced quantitative trading system that combines **statistical arbitrage**, **machine learning**, and **alternative data** (news sentiment) to generate alpha. This project demonstrates sophisticated skills in quantitative finance, AI/ML, and software engineering.

## 🎯 Project Overview

This system implements a **dynamic pairs trading strategy** with the following innovations:

- **Automated Pair Discovery**: Uses cointegration tests and clustering algorithms to identify trading pairs
- **ML-Powered Signals**: Gradient Boosting and XGBoost models predict spread mean reversion
- **Alternative Data**: News sentiment analysis using VADER and FinBERT (transformers)
- **Robust Backtesting**: Event-driven engine with transaction costs, slippage, and proper time-series handling
- **Professional Architecture**: PostgreSQL database, modular Python codebase, comprehensive testing

## 🏗️ Architecture

```
├── config/              # Configuration files
│   └── config.yaml     # Strategy parameters
├── data/               # Data storage
│   ├── raw/           # Raw market data
│   └── processed/     # Processed features
├── models/            # Trained ML models
├── results/           # Backtest results
├── plots/             # Performance visualizations
└── src/
    ├── data/          # Data collection & loading
    ├── database/      # PostgreSQL integration
    ├── pairs/         # Pair identification algorithms
    ├── sentiment/     # NLP sentiment analysis
    ├── features/      # Feature engineering
    ├── models/        # ML models (GB, XGBoost, LSTM)
    ├── backtest/      # Backtesting engine
    └── main.py        # Main pipeline
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PostgreSQL (optional, for full database features)
- API Keys (optional):
  - Alpha Vantage or Polygon.io (for market data)
  - NewsAPI (for news sentiment)

### Installation

1. **Clone and Setup**:
```bash
# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env

# Edit .env with your API keys (optional for basic demo)
nano .env
```

2. **Run the Pipeline**:
```bash
python src/main.py
```

This will:
1. Fetch S&P 500 stock data
2. Identify cointegrated pairs
3. Calculate spreads and z-scores
4. Train ML models
5. Run backtest with performance metrics

## 📊 Key Features

### 1. Pair Identification

**Cointegration Method** (Engle-Granger test):
```python
from src.pairs.pair_finder import CointegrationPairFinder

finder = CointegrationPairFinder(p_value_threshold=0.05)
pairs = finder.find_pairs(price_data)
```

**Clustering Method** (K-Means/DBSCAN):
```python
from src.pairs.pair_finder import ClusteringPairFinder

finder = ClusteringPairFinder(method='kmeans', n_clusters=20)
pairs = finder.find_pairs(price_data)
```

### 2. Sentiment Analysis

**VADER** (rule-based):
```python
from src.sentiment.sentiment_analyzer import SentimentAnalyzer

analyzer = SentimentAnalyzer(model_type='vader')
sentiment = analyzer.analyze_text("Apple reports record earnings!")
```

**FinBERT** (transformer model):
```python
analyzer = SentimentAnalyzer(model_type='finbert')
sentiment = analyzer.analyze_text("Stock market crashes amid recession fears")
```

### 3. Machine Learning

**Gradient Boosting**:
```python
from src.models.predictor import GradientBoostingPredictor

model = GradientBoostingPredictor(n_estimators=200, learning_rate=0.05)
model.train(X_train, y_train, X_val, y_val)
predictions = model.predict(X_test)
```

**XGBoost**:
```python
from src.models.predictor import XGBoostPredictor

model = XGBoostPredictor()
model.train(X_train, y_train, X_val, y_val)
```

### 4. Backtesting

**Event-Driven Engine**:
```python
from src.backtest.engine import BacktestEngine

engine = BacktestEngine(
    initial_capital=100000,
    transaction_cost=0.001,  # 0.1%
    slippage=0.0005,         # 0.05%
    entry_zscore_threshold=2.0
)

performance = engine.run_backtest(pairs_data, price_data, signals_data)
```

**Performance Metrics**:
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Win Rate
- Profit Factor

## 📈 Sample Results

```
PERFORMANCE METRICS
=============================================================
Returns:
  Total Return:           24.35%
  Annual Return:          11.82%

Risk-Adjusted:
  Sharpe Ratio:            1.84
  Sortino Ratio:           2.31
  Max Drawdown:            8.45%
  Volatility (annual):    12.30%

Trading:
  Number of Trades:         142
  Win Rate:               58.45%
  Profit Factor:           1.67
  Avg Trade Return:        0.85%
```

## 🛠️ Configuration

Edit `config/config.yaml` to customize:

```yaml
strategy:
  entry_zscore_threshold: 2.0    # Z-score to enter trade
  exit_zscore_threshold: 0.5     # Z-score to exit trade
  stop_loss_zscore: 3.5          # Stop loss threshold
  max_positions: 10              # Max concurrent positions

model:
  type: "gradient_boosting"      # or "xgboost" or "lightgbm"
  
backtesting:
  initial_capital: 100000
  transaction_cost: 0.001        # 0.1%
  slippage: 0.0005              # 0.05%
```

## 📚 Key Modules

### Data Collection (`src/data/data_loader.py`)
- Fetches historical price data (yfinance, Alpha Vantage, Polygon)
- Collects news articles (NewsAPI)
- Handles S&P 500 universe

### Pair Finding (`src/pairs/pair_finder.py`)
- Engle-Granger cointegration test
- K-Means and DBSCAN clustering
- Half-life calculation for mean reversion

### Sentiment Analysis (`src/sentiment/sentiment_analyzer.py`)
- VADER sentiment scoring
- FinBERT transformer model
- Feature engineering from sentiment

### Feature Engineering (`src/features/feature_engineer.py`)
- Spread-based features (z-scores, rolling stats)
- Technical indicators (SMA, RSI, volatility)
- Sentiment features (lags, momentum)

### ML Models (`src/models/predictor.py`)
- Gradient Boosting Regressor
- XGBoost
- LightGBM
- Feature importance analysis

### Backtesting (`src/backtest/engine.py`)
- Event-driven architecture
- Transaction costs and slippage
- Position management
- Stop loss handling

## 🧪 Testing

```bash
# Run unit tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

## 📊 Database Schema

The project includes a PostgreSQL schema for production deployment:

- `stock_prices`: Historical OHLCV data
- `news_articles`: News articles for sentiment
- `sentiment_scores`: Daily sentiment aggregates
- `stock_pairs`: Identified pairs with statistics
- `pair_spreads`: Spread time series
- `trading_signals`: ML model predictions
- `trades`: Executed trades
- `portfolio_performance`: Daily performance metrics

## 🎓 Learning Outcomes

This project demonstrates:

1. **Quantitative Finance**:
   - Statistical arbitrage strategies
   - Cointegration and mean reversion
   - Risk management and position sizing

2. **Machine Learning**:
   - Time series prediction
   - Feature engineering
   - Model evaluation and validation
   - Avoiding lookahead bias

3. **Natural Language Processing**:
   - Sentiment analysis
   - Transformer models (BERT)
   - Alternative data integration

4. **Software Engineering**:
   - Modular architecture
   - Database design
   - Event-driven systems
   - Professional documentation

## 🚧 Future Enhancements

- [ ] LSTM/GRU models for sequence prediction
- [ ] Real-time data integration
- [ ] Paper trading implementation
- [ ] Web dashboard (Streamlit/Flask)
- [ ] Advanced risk management
- [ ] Multi-timeframe analysis
- [ ] Portfolio optimization
- [ ] Cloud deployment (AWS/GCP)

## 📖 References

- **Cointegration**: Engle, R. F., & Granger, C. W. J. (1987). "Co-integration and error correction"
- **Pairs Trading**: Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006). "Pairs trading: Performance of a relative-value arbitrage rule"
- **FinBERT**: Araci, D. (2019). "FinBERT: Financial Sentiment Analysis with Pre-trained Language Models"

## 📄 License

MIT License - feel free to use this for learning and portfolio purposes!

## 👤 Author

**Ronald Jangam**

Built with ❤️ for quantitative finance and AI/ML enthusiasts.

---

⭐ **Star this repo** if you find it useful!

📧 Questions? Open an issue or reach out!