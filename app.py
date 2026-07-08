
from fastapi import FastAPI
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import xgboost as xgb
import os

app = FastAPI(title="Gold Price Prediction API")

# Load model with error handling
try:
    model = xgb.XGBRegressor()
    model.load_model('xgboost_gold_model.json')
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

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

@app.get("/")
def root():
    return {"message": "Gold Price Prediction API is running"}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.get("/predict")
def predict():
    if model is None:
        return {"error": "Model not loaded"}, 500
    
    try:
        end_date = datetime.now()
        start_date = end_date - pd.Timedelta(days=100)
        gold = yf.Ticker("GC=F")
        data = gold.history(start=start_date, end=end_date)
        
        if len(data) < 50:
            return {"error": "Not enough data available"}, 500
        
        data = calculate_indicators(data)
        latest = data.iloc[-1:].copy()
        
        features = ['SMA_20', 'SMA_50', 'EMA_20', 'RSI_14', 'MACD', 
                    'MACD_Signal', 'MACD_Histogram', 'BB_Upper', 'BB_Lower', 'BB_Middle']
        
        X = latest[features].values
        pred = model.predict(X)[0]
        current = latest['Close'].values[0]
        
        return {
            "current_price": round(float(current), 2),
            "predicted_price": round(float(pred), 2),
            "signal": "BUY" if pred > current else "SELL",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": str(e)}, 500