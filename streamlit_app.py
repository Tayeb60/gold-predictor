import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import xgboost as xgb
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Gold Price Predictor",
    page_icon="📈",
    layout="wide"
)

# Title
st.title("📈 Gold Price Prediction Dashboard")
st.markdown("**AI-powered gold price prediction and trading signals**")

# Load model
@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    model.load_model('xgboost_gold_model.json')
    return model

model = load_model()

# Function to calculate indicators
def calculate_indicators(df):
    df = df.copy()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    return df

# Function to get prediction
def get_prediction():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=100)
    gold = yf.Ticker("GC=F")
    data = gold.history(start=start_date, end=end_date)
    
    if len(data) < 50:
        return None, None, None, None
    
    data = calculate_indicators(data)
    latest = data.iloc[-1:].copy()
    
    features = ['SMA_20', 'SMA_50', 'EMA_20', 'RSI_14', 'MACD', 
                'MACD_Signal', 'MACD_Histogram', 'BB_Upper', 'BB_Lower', 'BB_Middle']
    
    X = latest[features].values
    pred = model.predict(X)[0]
    current = latest['Close'].values[0]
    
    return current, pred, data, latest

# Get prediction
current_price, predicted_price, historical_data, latest_data = get_prediction()

if current_price is None:
    st.error("❌ Not enough data to make a prediction. Please try again later.")
    st.stop()

# Layout: Two columns
col1, col2 = st.columns(2)

with col1:
    st.metric(
        label="Current Gold Price",
        value=f"${current_price:,.2f}",
        delta=None
    )
    
    st.metric(
        label="Predicted Price (Next Day)",
        value=f"${predicted_price:,.2f}",
        delta=f"${predicted_price - current_price:,.2f}",
        delta_color="normal"
    )
    
    # Signal
    signal = "BUY" if predicted_price > current_price else "SELL"
    color = "green" if signal == "BUY" else "red"
    st.markdown(f"""
    <div style="background-color:{color}; padding:20px; border-radius:10px; text-align:center;">
        <h2 style="color:white;">{signal}</h2>
        <p style="color:white;">Trading Signal</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Confidence
    confidence = abs(predicted_price - current_price) / current_price * 100
    st.metric(
        label="Confidence",
        value=f"{confidence:.2f}%",
        delta=None
    )

with col2:
    # Price chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=historical_data.index,
        y=historical_data['Close'],
        mode='lines',
        name='Actual Price',
        line=dict(color='gold', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=[historical_data.index[-1], historical_data.index[-1] + timedelta(days=1)],
        y=[current_price, predicted_price],
        mode='lines+markers',
        name='Predicted Price',
        line=dict(color='blue', width=2, dash='dash'),
        marker=dict(size=10)
    ))
    
    fig.update_layout(
        title='Gold Price: Actual vs Predicted',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        height=400,
        template='plotly_dark'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Technical Indicators
st.subheader("📊 Technical Indicators")
st.markdown("Key technical indicators used by the model")

col3, col4, col5 = st.columns(3)

with col3:
    latest_rsi = latest_data['RSI_14'].values[0] if not pd.isna(latest_data['RSI_14'].values[0]) else 50
    st.metric(
        label="RSI (14)",
        value=f"{latest_rsi:.2f}",
        delta="Overbought" if latest_rsi > 70 else "Oversold" if latest_rsi < 30 else "Neutral"
    )

with col4:
    latest_macd = latest_data['MACD'].values[0] if not pd.isna(latest_data['MACD'].values[0]) else 0
    latest_signal = latest_data['MACD_Signal'].values[0] if not pd.isna(latest_data['MACD_Signal'].values[0]) else 0
    macd_signal = "Bullish" if latest_macd > latest_signal else "Bearish"
    st.metric(
        label="MACD",
        value=f"{latest_macd:.2f}",
        delta=macd_signal
    )

with col5:
    latest_bb_upper = latest_data['BB_Upper'].values[0] if not pd.isna(latest_data['BB_Upper'].values[0]) else current_price
    latest_bb_lower = latest_data['BB_Lower'].values[0] if not pd.isna(latest_data['BB_Lower'].values[0]) else current_price
    bb_position = "Upper Band" if current_price > latest_bb_upper else "Lower Band" if current_price < latest_bb_lower else "Middle"
    st.metric(
        label="Bollinger Bands",
        value=f"${current_price:,.2f}",
        delta=bb_position
    )
