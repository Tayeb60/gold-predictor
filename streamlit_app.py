import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import xgboost as xgb
import plotly.graph_objects as go
import requests
import time

# ----------------------------------------------
# Page configuration
# ----------------------------------------------
st.set_page_config(
    page_title="Gold Price Predictor",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Gold Price Prediction Dashboard")
st.markdown("**AI-powered gold price prediction and trading signals**")

# ----------------------------------------------
# Force refresh using query param
# ----------------------------------------------
if "refresh" in st.query_params:
    st.cache_data.clear()
    st.cache_resource.clear()
    # Remove the query param to avoid infinite loop
    st.query_params.clear()

# ----------------------------------------------
# Manual refresh button with full page reload
# ----------------------------------------------
col_refresh = st.columns([1, 5])
with col_refresh[0]:
    if st.button("🔄 Refresh Data"):
        # Use a query param to trigger a full page reload with cache clear
        st.markdown(
            '<meta http-equiv="refresh" content="0; url=?refresh=true">',
            unsafe_allow_html=True
        )
        st.stop()

# ----------------------------------------------
# Exchange rate function (no caching)
# ----------------------------------------------
def get_usd_to_gbp():
    """Fetch live USD to GBP exchange rate."""
    try:
        url = "https://api.frankfurter.app/latest?from=USD&to=GBP"
        response = requests.get(url)
        data = response.json()
        return data['rates']['GBP']
    except Exception as e:
        st.warning(f"Exchange rate API failed: {e}. Using fallback rate 0.80.")
        return 0.80

# ----------------------------------------------
# Load model
# ----------------------------------------------
@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    model.load_model('xgboost_gold_model.json')
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ----------------------------------------------
# Technical indicator function
# ----------------------------------------------
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

# ----------------------------------------------
# Fetch data and make prediction (NO CACHING)
# ----------------------------------------------
def fetch_gold_data():
    """Fetch fresh gold data from Yahoo Finance (no caching)."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=100)
    gold = yf.Ticker("GC=F")
    
    # Force fresh data
    data = gold.history(
        start=start_date, 
        end=end_date, 
        interval="1d",
        auto_adjust=False
    )
    
    # Remove duplicate indices if any
    data = data[~data.index.duplicated(keep='last')]
    
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

# ----------------------------------------------
# Display last update time
# ----------------------------------------------
update_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
st.caption(f"🕒 Last updated: {update_time}")

# ----------------------------------------------
# Get exchange rate
# ----------------------------------------------
usd_to_gbp = get_usd_to_gbp()

# ----------------------------------------------
# Get prediction with fresh data
# ----------------------------------------------
current_price, predicted_price, historical_data, latest_data = fetch_gold_data()

if current_price is None:
    st.error("❌ Not enough data to make a prediction. Please try again later.")
    st.stop()

# Convert prices to GBP
current_price_gbp = current_price * usd_to_gbp
predicted_price_gbp = predicted_price * usd_to_gbp

# ----------------------------------------------
# Layout: Two columns for metrics
# ----------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.metric(
        label="Current Gold Price (GBP)",
        value=f"£{current_price_gbp:,.2f}",
        delta=None
    )
    
    st.metric(
        label="Predicted Price (Next Day)",
        value=f"£{predicted_price_gbp:,.2f}",
        delta=f"£{predicted_price_gbp - current_price_gbp:,.2f}",
        delta_color="normal"
    )

with col2:
    # Signal
    signal = "BUY" if predicted_price > current_price else "SELL"
    color = "green" if signal == "BUY" else "red"
    st.markdown(f"""
    <div style="background-color:{color}; padding:20px; border-radius:10px; text-align:center;">
        <h2 style="color:white;">{signal}</h2>
        <p style="color:white;">Trading Signal</p>
    </div>
    """, unsafe_allow_html=True)
    
    confidence = abs(predicted_price - current_price) / current_price * 100
    st.metric(
        label="Confidence",
        value=f"{confidence:.2f}%",
        delta=None
    )

# ----------------------------------------------
# Price chart
# ----------------------------------------------
st.subheader("📊 Gold Price: Actual vs Predicted")

fig = go.Figure()

# Historical prices in GBP
historical_close_gbp = historical_data['Close'] * usd_to_gbp

fig.add_trace(go.Scatter(
    x=historical_data.index,
    y=historical_close_gbp,
    mode='lines',
    name='Actual Price',
    line=dict(color='gold', width=2)
))

# Predicted price (next day)
fig.add_trace(go.Scatter(
    x=[historical_data.index[-1], historical_data.index[-1] + timedelta(days=1)],
    y=[current_price_gbp, predicted_price_gbp],
    mode='lines+markers',
    name='Predicted Price',
    line=dict(color='blue', width=2, dash='dash'),
    marker=dict(size=10)
))

fig.update_layout(
    xaxis_title='Date',
    yaxis_title='Price (GBP)',
    height=400,
    template='plotly_dark'
)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------------------------
# Technical Indicators
# ----------------------------------------------
st.subheader("📊 Technical Indicators")
st.markdown("Key technical indicators used by the model")

col3, col4, col5 = st.columns(3)

with col3:
    latest_rsi = latest_data['RSI_14'].values[0] if not pd.isna(latest_data['RSI_14'].values[0]) else 50
    rsi_status = "Overbought" if latest_rsi > 70 else "Oversold" if latest_rsi < 30 else "Neutral"
    st.metric(
        label="RSI (14)",
        value=f"{latest_rsi:.2f}",
        delta=rsi_status
    )

with col4:
    latest_macd = latest_data['MACD'].values[0] if not pd.isna(latest_data['MACD'].values[0]) else 0
    latest_signal = latest_data['MACD_Signal'].values[0] if not pd.isna(latest_data['MACD_Signal'].values[0]) else 0
    macd_status = "Bullish" if latest_macd > latest_signal else "Bearish"
    st.metric(
        label="MACD",
        value=f"{latest_macd:.2f}",
        delta=macd_status
    )

with col5:
    latest_bb_upper = latest_data['BB_Upper'].values[0] if not pd.isna(latest_data['BB_Upper'].values[0]) else current_price
    latest_bb_lower = latest_data['BB_Lower'].values[0] if not pd.isna(latest_data['BB_Lower'].values[0]) else current_price
    bb_position = "Upper Band" if current_price > latest_bb_upper else "Lower Band" if current_price < latest_bb_lower else "Middle"
    st.metric(
        label="Bollinger Bands",
        value=f"£{current_price_gbp:,.2f}",
        delta=bb_position
    )

# ----------------------------------------------
# Footer
# ----------------------------------------------
st.markdown("---")
st.caption("Data sourced from Yahoo Finance. Predictions are for educational purposes only. Not financial advice.")