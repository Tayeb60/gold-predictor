import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

silver_ticker = "SI=F"
end_date = datetime.now()
start_date = end_date - timedelta(days=5*365)

silver = yf.Ticker(silver_ticker)
data = silver.history(start=start_date, end=end_date)
data.to_csv('silver_price_data.csv')

print("Silver data saved to 'silver_price_data.csv'")