"""
Streamlit Dashboard for AI-Powered Pairs Trading Strategy
Visualizes strategy performance, pair analysis, and sentiment insights
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtest.metrics import calculate_sharpe_ratio, calculate_max_drawdown, calculate_sortino_ratio


st.set_page_config(
    page_title="AI Pairs Trading Dashboard",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("🤖 AI-Powered Pairs Trading Dashboard")
st.markdown("**Dynamic Pairs Trading with News Sentiment Analysis**")

# Sidebar
st.sidebar.header("Strategy Controls")
strategy_type = st.sidebar.selectbox(
    "Select Strategy",
    ["ML-Enhanced Pairs Trading", "Traditional Pairs Trading", "Sentiment-Only"]
)

lookback_period = st.sidebar.slider("Lookback Period (days)", 30, 365, 60)
z_score_threshold = st.sidebar.slider("Z-Score Entry Threshold", 1.0, 3.0, 2.0, 0.1)

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Performance", "🔗 Top Pairs", "📰 Sentiment", "⚙️ Analytics"])

with tab1:
    st.header("Strategy Performance")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Generate sample performance data
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
    cumulative_returns = np.cumsum(np.random.randn(252) * 0.01 + 0.0005)
    
    # Calculate metrics
    returns = np.diff(cumulative_returns, prepend=0)
    sharpe = calculate_sharpe_ratio(returns)
    max_dd = calculate_max_drawdown(cumulative_returns)
    sortino = calculate_sortino_ratio(returns)
    total_return = cumulative_returns[-1] * 100
    
    with col1:
        st.metric("Total Return", f"{total_return:.2f}%", delta=f"{total_return/12:.2f}%")
    with col2:
        st.metric("Sharpe Ratio", f"{sharpe:.2f}", delta="0.3")
    with col3:
        st.metric("Max Drawdown", f"{max_dd:.2f}%", delta="-2.1%")
    with col4:
        st.metric("Sortino Ratio", f"{sortino:.2f}", delta="0.4")
    
    # Equity curve
    st.subheader("Equity Curve")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, 
        y=cumulative_returns * 100,
        mode='lines',
        name='Strategy',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Add benchmark
    benchmark_returns = np.cumsum(np.random.randn(252) * 0.008 + 0.0003)
    fig.add_trace(go.Scatter(
        x=dates,
        y=benchmark_returns * 100,
        mode='lines',
        name='S&P 500 Benchmark',
        line=dict(color='#ff7f0e', width=2, dash='dash')
    ))
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Cumulative Return (%)",
        hovermode='x unified',
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Monthly returns heatmap
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Monthly Returns (%)")
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        monthly_data = np.random.randn(2, 12) * 2 + 1
        
        fig = go.Figure(data=go.Heatmap(
            z=monthly_data,
            x=months,
            y=['2024', '2025'],
            colorscale='RdYlGn',
            text=monthly_data,
            texttemplate='%{text:.1f}',
            textfont={"size": 10},
        ))
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Trade Statistics")
        trade_stats = pd.DataFrame({
            'Metric': ['Total Trades', 'Win Rate', 'Avg Win', 'Avg Loss', 'Profit Factor'],
            'Value': ['147', '58.5%', '2.3%', '-1.2%', '1.85']
        })
        st.dataframe(trade_stats, hide_index=True, use_container_width=True)

with tab2:
    st.header("Top Performing Pairs")
    
    # Sample pairs data
    pairs_data = pd.DataFrame({
        'Pair': ['AAPL-MSFT', 'JPM-BAC', 'KO-PEP', 'XOM-CVX', 'WMT-TGT'],
        'Cointegration Score': [0.89, 0.85, 0.92, 0.78, 0.81],
        'Current Z-Score': [2.1, -1.8, 0.5, 2.5, -0.3],
        'YTD Return': ['12.5%', '8.3%', '15.2%', '6.7%', '9.1%'],
        'Trades': [23, 18, 31, 15, 20],
        'Status': ['Long', 'Short', 'Neutral', 'Long', 'Neutral']
    })
    
    st.dataframe(
        pairs_data.style.applymap(
            lambda x: 'background-color: #90EE90' if x == 'Long' 
            else ('background-color: #FFB6C6' if x == 'Short' else ''),
            subset=['Status']
        ),
        hide_index=True,
        use_container_width=True
    )
    
    # Pair spread visualization
    st.subheader("Pair Spread Analysis: AAPL-MSFT")
    
    dates_spread = pd.date_range(end=datetime.now(), periods=180, freq='D')
    spread = np.cumsum(np.random.randn(180) * 0.5)
    z_score = (spread - spread.mean()) / spread.std()
    
    fig = go.Figure()
    
    # Z-score plot
    fig.add_trace(go.Scatter(
        x=dates_spread,
        y=z_score,
        mode='lines',
        name='Z-Score',
        line=dict(color='blue', width=2)
    ))
    
    # Entry/exit thresholds
    fig.add_hline(y=2.0, line_dash="dash", line_color="red", annotation_text="Entry Threshold")
    fig.add_hline(y=-2.0, line_dash="dash", line_color="red")
    fig.add_hline(y=0, line_dash="dot", line_color="gray", annotation_text="Mean")
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Z-Score",
        hovermode='x unified',
        height=350
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Sentiment Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Sentiment Trends")
        
        dates_sentiment = pd.date_range(end=datetime.now(), periods=90, freq='D')
        sentiment_aapl = np.random.randn(90) * 0.2 + 0.1
        sentiment_msft = np.random.randn(90) * 0.2 + 0.05
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates_sentiment,
            y=sentiment_aapl,
            mode='lines',
            name='AAPL Sentiment',
            line=dict(color='#00B0F0', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=dates_sentiment,
            y=sentiment_msft,
            mode='lines',
            name='MSFT Sentiment',
            line=dict(color='#92D050', width=2)
        ))
        
        fig.add_hline(y=0, line_dash="dot", line_color="gray")
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Sentiment Score",
            hovermode='x unified',
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Latest News")
        news_items = [
            {"stock": "AAPL", "headline": "Apple announces new AI features", "sentiment": 0.75, "time": "2h ago"},
            {"stock": "MSFT", "headline": "Microsoft Cloud revenue beats estimates", "sentiment": 0.82, "time": "4h ago"},
            {"stock": "AAPL", "headline": "iPhone sales meet expectations", "sentiment": 0.45, "time": "6h ago"},
            {"stock": "MSFT", "headline": "Azure growth slows slightly", "sentiment": -0.15, "time": "8h ago"},
        ]
        
        for item in news_items:
            sentiment_color = "🟢" if item['sentiment'] > 0.3 else ("🔴" if item['sentiment'] < -0.3 else "🟡")
            st.markdown(f"""
            **{item['stock']}** {sentiment_color}  
            {item['headline']}  
            *{item['time']} • Score: {item['sentiment']:.2f}*
            """)
            st.divider()

with tab4:
    st.header("Advanced Analytics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Feature Importance")
        features = ['Sentiment Diff', 'Z-Score', 'Volume Ratio', 'Volatility', 'RSI Diff', 'MA Cross']
        importance = [0.28, 0.24, 0.18, 0.14, 0.10, 0.06]
        
        fig = go.Figure(go.Bar(
            x=importance,
            y=features,
            orientation='h',
            marker=dict(color=importance, colorscale='Viridis')
        ))
        fig.update_layout(
            xaxis_title="Importance Score",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Model Performance")
        model_data = pd.DataFrame({
            'Model': ['LSTM', 'Gradient Boosting', 'Random Forest', 'Linear Regression'],
            'Accuracy': [0.67, 0.71, 0.65, 0.58],
            'Sharpe': [1.45, 1.82, 1.31, 0.95]
        })
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=model_data['Accuracy'],
            y=model_data['Sharpe'],
            mode='markers+text',
            text=model_data['Model'],
            textposition="top center",
            marker=dict(size=15, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ))
        fig.update_layout(
            xaxis_title="Prediction Accuracy",
            yaxis_title="Sharpe Ratio",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Rolling metrics
    st.subheader("Rolling Performance Metrics (30-Day Window)")
    
    dates_rolling = pd.date_range(end=datetime.now(), periods=180, freq='D')
    rolling_sharpe = np.random.randn(180) * 0.3 + 1.5
    rolling_vol = np.random.randn(180) * 0.02 + 0.12
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates_rolling,
        y=rolling_sharpe,
        mode='lines',
        name='Rolling Sharpe Ratio',
        yaxis='y',
        line=dict(color='blue', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=dates_rolling,
        y=rolling_vol * 100,
        mode='lines',
        name='Rolling Volatility (%)',
        yaxis='y2',
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis=dict(title="Sharpe Ratio", side='left'),
        yaxis2=dict(title="Volatility (%)", side='right', overlaying='y'),
        hovermode='x unified',
        height=350
    )
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.info("""
**Project**: AI-Powered Pairs Trading  
**Features**: 
- Machine Learning Predictions
- News Sentiment Analysis  
- Event-Driven Backtesting
- Real-time Performance Metrics
""")
