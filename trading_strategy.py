import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import xgboost as xgb

model = xgb.XGBRegressor()
model.load_model('xgboost_gold_model.json')

def get_signal():
    # Fetch data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)
    gold = yf.Ticker("GC=F")
    data = gold.history(start=start_date, end=end_date)
    
    # Calculate indicators
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
    
    # RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI_14'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
    
    data['BB_Middle'] = data['Close'].rolling(window=20).mean()
    bb_std = data['Close'].rolling(window=20).std()
    data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
    data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
    
    # Get latest
    latest = data.iloc[-1:].copy()
    features = ['SMA_20', 'SMA_50', 'EMA_20', 'RSI_14', 'MACD', 
                'MACD_Signal', 'MACD_Histogram', 'BB_Upper', 'BB_Lower', 'BB_Middle']
    X = latest[features].values
    
    # Predict
    pred = model.predict(X)[0]
    current = latest['Close'].values[0]
    
    # Generate signal
    signal = "BUY" if pred > current else "SELL"
    confidence = abs(pred - current) / current * 100
    
    return {
        'signal': signal,
        'current_price': current,
        'predicted_price': pred,
        'confidence': confidence,
        'timestamp': datetime.now().isoformat()
    }

if __name__ == "__main__":
    result = get_signal()
    print(f"Signal: {result['signal']}")
    print(f"Current Price: ${result['current_price']:.2f}")
    print(f"Predicted Price: ${result['predicted_price']:.2f}")
    print(f"Confidence: {result['confidence']:.2f}%")