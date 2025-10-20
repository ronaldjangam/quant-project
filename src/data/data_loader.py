"""
Data loading and fetching utilities for stock prices and news.
"""

import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import pandas as pd
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
from newsapi import NewsApiClient
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()


class StockDataLoader:
    """Fetch historical stock price data."""
    
    def __init__(self, data_source: str = "yfinance"):
        """
        Initialize data loader.
        
        Args:
            data_source: 'yfinance', 'alpha_vantage', or 'polygon'
        """
        self.data_source = data_source
        
        if data_source == "alpha_vantage":
            self.av_key = os.getenv('ALPHA_VANTAGE_API_KEY')
            self.ts = TimeSeries(key=self.av_key, output_format='pandas')
        elif data_source == "polygon":
            self.polygon_key = os.getenv('POLYGON_API_KEY')
    
    def fetch_stock_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch historical stock data for a single ticker.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with OHLCV data
        """
        if self.data_source == "yfinance":
            return self._fetch_yfinance(ticker, start_date, end_date)
        elif self.data_source == "alpha_vantage":
            return self._fetch_alpha_vantage(ticker)
        else:
            raise ValueError(f"Unsupported data source: {self.data_source}")
    
    def _fetch_yfinance(
        self,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """Fetch data using yfinance."""
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            
            if df.empty:
                print(f"⚠ No data found for {ticker}")
                return pd.DataFrame()
            
            # Standardize column names
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Add adj_close (yfinance already adjusts 'Close')
            df['adj_close'] = df['close']
            df['ticker'] = ticker
            df.reset_index(inplace=True)
            df.rename(columns={'Date': 'date'}, inplace=True)
            
            return df[['ticker', 'date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']]
            
        except Exception as e:
            print(f"✗ Error fetching data for {ticker}: {e}")
            return pd.DataFrame()
    
    def _fetch_alpha_vantage(self, ticker: str) -> pd.DataFrame:
        """Fetch data using Alpha Vantage."""
        try:
            data, meta_data = self.ts.get_daily_adjusted(symbol=ticker, outputsize='full')
            
            df = data.copy()
            df.columns = ['open', 'high', 'low', 'close', 'adj_close', 'volume', 'dividend', 'split']
            df = df[['open', 'high', 'low', 'close', 'adj_close', 'volume']]
            df['ticker'] = ticker
            df.reset_index(inplace=True)
            df.rename(columns={'date': 'date'}, inplace=True)
            
            return df
            
        except Exception as e:
            print(f"✗ Error fetching data for {ticker}: {e}")
            return pd.DataFrame()
    
    def fetch_multiple_tickers(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Fetch data for multiple tickers.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            show_progress: Show progress bar
            
        Returns:
            Combined DataFrame with all tickers
        """
        all_data = []
        
        iterator = tqdm(tickers, desc="Fetching stock data") if show_progress else tickers
        
        for ticker in iterator:
            df = self.fetch_stock_data(ticker, start_date, end_date)
            if not df.empty:
                all_data.append(df)
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        else:
            return pd.DataFrame()


class NewsDataLoader:
    """Fetch news articles for stocks."""
    
    def __init__(self):
        """Initialize news API client."""
        self.api_key = os.getenv('NEWS_API_KEY')
        if self.api_key:
            self.newsapi = NewsApiClient(api_key=self.api_key)
        else:
            self.newsapi = None
            print("⚠ NEWS_API_KEY not found. News fetching disabled.")
    
    def fetch_news(
        self,
        ticker: str,
        company_name: str,
        start_date: str,
        end_date: str,
        language: str = 'en'
    ) -> List[Dict]:
        """
        Fetch news articles for a ticker.
        
        Args:
            ticker: Stock ticker symbol
            company_name: Company name for better search
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            language: News language
            
        Returns:
            List of article dictionaries
        """
        if not self.newsapi:
            return []
        
        try:
            # Search query
            query = f"{ticker} OR {company_name}"
            
            # Fetch articles
            articles = self.newsapi.get_everything(
                q=query,
                from_param=start_date,
                to=end_date,
                language=language,
                sort_by='publishedAt',
                page_size=100
            )
            
            # Process articles
            processed_articles = []
            for article in articles.get('articles', []):
                processed_articles.append({
                    'ticker': ticker,
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'content': article.get('content', ''),
                    'url': article.get('url', ''),
                    'published_at': article.get('publishedAt', ''),
                    'source': article.get('source', {}).get('name', '')
                })
            
            return processed_articles
            
        except Exception as e:
            print(f"✗ Error fetching news for {ticker}: {e}")
            return []
    
    def fetch_news_batch(
        self,
        tickers: List[str],
        company_names: Dict[str, str],
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        Fetch news for multiple tickers.
        
        Args:
            tickers: List of ticker symbols
            company_names: Dict mapping ticker to company name
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with all articles
        """
        all_articles = []
        
        for ticker in tqdm(tickers, desc="Fetching news"):
            company_name = company_names.get(ticker, ticker)
            articles = self.fetch_news(ticker, company_name, start_date, end_date)
            all_articles.extend(articles)
        
        if all_articles:
            return pd.DataFrame(all_articles)
        else:
            return pd.DataFrame()


def get_sp500_tickers() -> List[str]:
    """
    Get list of S&P 500 tickers.
    
    Returns:
        List of ticker symbols
    """
    try:
        # Fetch from Wikipedia
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        tables = pd.read_html(url)
        df = tables[0]
        tickers = df['Symbol'].tolist()
        
        # Clean tickers (remove dots for yfinance compatibility)
        tickers = [ticker.replace('.', '-') for ticker in tickers]
        
        print(f"✓ Loaded {len(tickers)} S&P 500 tickers")
        return tickers
        
    except Exception as e:
        print(f"✗ Error fetching S&P 500 tickers: {e}")
        return []


if __name__ == "__main__":
    # Test data loading
    loader = StockDataLoader(data_source="yfinance")
    
    # Test single ticker
    df = loader.fetch_stock_data("AAPL", "2023-01-01", "2024-01-01")
    print(f"\nAAPL data shape: {df.shape}")
    print(df.head())
    
    # Test multiple tickers
    tickers = ["AAPL", "MSFT", "GOOGL"]
    df_multi = loader.fetch_multiple_tickers(tickers, "2023-01-01", "2024-01-01")
    print(f"\nMultiple tickers data shape: {df_multi.shape}")
    print(df_multi.groupby('ticker').size())
