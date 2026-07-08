import numpy as np
import pandas as pd
import xgboost as xgb
from datetime import datetime, timedelta
import yfinance as yf

# Load the trained model
model = xgb.XGBRegressor()
model.load_model('xgboost_gold_model.json')

# Function to fetch latest gold data
def fetch_latest_gold_data(days=60):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    gold = yf.Ticker("GC=F")
    data = gold.history(start=start_date, end=end_date)
    return data

# Function to calculate technical indicators
def calculate_indicators(df):
    df = df.copy()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    return df

# Function to predict next day's price
def predict_next_price():
    # Fetch latest data
    data = fetch_latest_gold_data(60)
    
    # Calculate indicators
    data = calculate_indicators(data)
    
    # Get the latest row
    latest = data.iloc[-1:].copy()
    
    # Select features
    feature_columns = ['SMA_20', 'SMA_50', 'EMA_20', 'RSI_14', 'MACD', 
                       'MACD_Signal', 'MACD_Histogram', 'BB_Upper', 'BB_Lower', 'BB_Middle']
    
    # Prepare input
    X = latest[feature_columns].values
    
    # Make prediction
    prediction = model.predict(X)[0]
    
    return prediction, latest['Close'].values[0]

# Run prediction
predicted_price, current_price = predict_next_price()
print(f"Current Gold Price: ${current_price:.2f}")
print(f"Predicted Gold Price (next day): ${predicted_price:.2f}")
print(f"Predicted Direction: {'UP' if predicted_price > current_price else 'DOWN'}")