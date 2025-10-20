"""
Sentiment analysis using VADER and transformer models.
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict
from datetime import datetime
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from tqdm import tqdm

# Download VADER lexicon if needed
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)


class SentimentAnalyzer:
    """Perform sentiment analysis on news articles."""
    
    def __init__(self, model_type: str = 'vader'):
        """
        Initialize sentiment analyzer.
        
        Args:
            model_type: 'vader' or 'finbert'
        """
        self.model_type = model_type
        
        if model_type == 'vader':
            self.analyzer = SentimentIntensityAnalyzer()
            print("✓ Initialized VADER sentiment analyzer")
            
        elif model_type == 'finbert':
            # FinBERT: Pre-trained BERT for financial sentiment
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "ProsusAI/finbert"
                )
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    "ProsusAI/finbert"
                )
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.model.to(self.device)
                print(f"✓ Initialized FinBERT on {self.device}")
            except Exception as e:
                print(f"✗ Error loading FinBERT: {e}")
                print("Falling back to VADER")
                self.model_type = 'vader'
                self.analyzer = SentimentIntensityAnalyzer()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def analyze_text_vader(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using VADER.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        if not text or not isinstance(text, str):
            return {
                'compound': 0.0,
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0
            }
        
        scores = self.analyzer.polarity_scores(text)
        
        return {
            'compound': scores['compound'],
            'positive': scores['pos'],
            'negative': scores['neg'],
            'neutral': scores['neu']
        }
    
    def analyze_text_finbert(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment using FinBERT.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        if not text or not isinstance(text, str):
            return {
                'compound': 0.0,
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0
            }
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # FinBERT outputs: [negative, neutral, positive]
            probs = probs.cpu().numpy()[0]
            
            # Convert to compound score (-1 to 1)
            compound = probs[2] - probs[0]
            
            return {
                'compound': float(compound),
                'positive': float(probs[2]),
                'negative': float(probs[0]),
                'neutral': float(probs[1])
            }
            
        except Exception as e:
            print(f"Error in FinBERT analysis: {e}")
            return {
                'compound': 0.0,
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0
            }
    
    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze text sentiment using configured model.
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment scores
        """
        if self.model_type == 'vader':
            return self.analyze_text_vader(text)
        else:
            return self.analyze_text_finbert(text)
    
    def analyze_articles(
        self,
        articles: pd.DataFrame,
        text_column: str = 'title'
    ) -> pd.DataFrame:
        """
        Analyze sentiment for multiple articles.
        
        Args:
            articles: DataFrame with articles
            text_column: Column containing text to analyze
            
        Returns:
            DataFrame with sentiment scores added
        """
        results = []
        
        for _, row in tqdm(articles.iterrows(), total=len(articles), desc="Analyzing sentiment"):
            text = row.get(text_column, '')
            
            # Optionally combine title and description
            if 'description' in row and pd.notna(row['description']):
                text = f"{text} {row['description']}"
            
            sentiment = self.analyze_text(text)
            
            results.append({
                **row.to_dict(),
                'sentiment_compound': sentiment['compound'],
                'sentiment_positive': sentiment['positive'],
                'sentiment_negative': sentiment['negative'],
                'sentiment_neutral': sentiment['neutral']
            })
        
        return pd.DataFrame(results)
    
    def aggregate_daily_sentiment(
        self,
        articles: pd.DataFrame,
        date_column: str = 'published_at'
    ) -> pd.DataFrame:
        """
        Aggregate sentiment scores by ticker and date.
        
        Args:
            articles: DataFrame with sentiment scores
            date_column: Column containing date
            
        Returns:
            DataFrame with daily aggregated sentiment
        """
        # Ensure date column is datetime
        if articles[date_column].dtype == 'object':
            articles[date_column] = pd.to_datetime(articles[date_column])
        
        # Extract date
        articles['date'] = articles[date_column].dt.date
        
        # Aggregate by ticker and date
        daily_sentiment = articles.groupby(['ticker', 'date']).agg({
            'sentiment_compound': 'mean',
            'sentiment_positive': 'mean',
            'sentiment_negative': 'mean',
            'sentiment_neutral': 'mean',
            'title': 'count'  # Count articles
        }).reset_index()
        
        daily_sentiment.rename(columns={'title': 'article_count'}, inplace=True)
        
        return daily_sentiment


class SentimentFeatureEngineer:
    """Create sentiment-based features for ML models."""
    
    @staticmethod
    def create_lag_features(
        sentiment_df: pd.DataFrame,
        lags: List[int] = [1, 2, 3, 5, 7]
    ) -> pd.DataFrame:
        """
        Create lagged sentiment features.
        
        Args:
            sentiment_df: DataFrame with sentiment scores
            lags: List of lag periods
            
        Returns:
            DataFrame with lag features
        """
        df = sentiment_df.copy()
        df = df.sort_values('date')
        
        for lag in lags:
            df[f'sentiment_lag_{lag}'] = df.groupby('ticker')['sentiment_compound'].shift(lag)
        
        return df
    
    @staticmethod
    def create_rolling_features(
        sentiment_df: pd.DataFrame,
        windows: List[int] = [3, 7, 14]
    ) -> pd.DataFrame:
        """
        Create rolling sentiment features.
        
        Args:
            sentiment_df: DataFrame with sentiment scores
            windows: List of window sizes
            
        Returns:
            DataFrame with rolling features
        """
        df = sentiment_df.copy()
        df = df.sort_values('date')
        
        for window in windows:
            # Rolling mean
            df[f'sentiment_ma_{window}'] = (
                df.groupby('ticker')['sentiment_compound']
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
            
            # Rolling std
            df[f'sentiment_std_{window}'] = (
                df.groupby('ticker')['sentiment_compound']
                .rolling(window=window, min_periods=1)
                .std()
                .reset_index(level=0, drop=True)
            )
        
        return df
    
    @staticmethod
    def create_momentum_features(
        sentiment_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create sentiment momentum features.
        
        Args:
            sentiment_df: DataFrame with sentiment scores
            
        Returns:
            DataFrame with momentum features
        """
        df = sentiment_df.copy()
        df = df.sort_values('date')
        
        # Sentiment change
        df['sentiment_change'] = df.groupby('ticker')['sentiment_compound'].diff()
        
        # Sentiment acceleration
        df['sentiment_acceleration'] = df.groupby('ticker')['sentiment_change'].diff()
        
        return df


if __name__ == "__main__":
    # Test sentiment analysis
    print("Sentiment Analysis Module")
    print("-" * 50)
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer(model_type='vader')
    
    # Test texts
    texts = [
        "Apple stock surges to all-time high on strong earnings",
        "Company reports disappointing quarterly results",
        "Tech stocks show mixed performance in volatile market"
    ]
    
    print("\nTest Sentiment Analysis:")
    for text in texts:
        sentiment = analyzer.analyze_text(text)
        print(f"\nText: {text}")
        print(f"Compound: {sentiment['compound']:.3f}")
        print(f"Positive: {sentiment['positive']:.3f}")
        print(f"Negative: {sentiment['negative']:.3f}")
