import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import xgboost as xgb

print("Loading model...")
model = xgb.XGBRegressor()
model.load_model('xgboost_gold_model.json')

print("Fetching data...")
end_date = datetime.now()
start_date = end_date - timedelta(days=60)
gold = yf.Ticker("GC=F")
data = gold.history(start=start_date, end=end_date)

print(f"Data shape: {data.shape}")
print(f"Columns: {data.columns.tolist()}")

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

print("Calculating indicators...")
data = calculate_indicators(data)
latest = data.iloc[-1:].copy()

features = ['SMA_20', 'SMA_50', 'EMA_20', 'RSI_14', 'MACD', 
            'MACD_Signal', 'MACD_Histogram', 'BB_Upper', 'BB_Lower', 'BB_Middle']

print("Features available:", [col for col in features if col in data.columns])
missing = [col for col in features if col not in data.columns]
if missing:
    print(f"Missing features: {missing}")

X = latest[features].values
print(f"Input shape: {X.shape}")
print(f"Input values: {X}")

print("Making prediction...")
pred = model.predict(X)[0]
current = latest['Close'].values[0]

print(f"Current Price: ${current:.2f}")
print(f"Predicted Price: ${pred:.2f}")
print(f"Signal: {'BUY' if pred > current else 'SELL'}")