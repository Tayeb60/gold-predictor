import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Define the gold ticker symbol (GC=F for gold futures, GLD for Gold ETF)
gold_ticker = "GC=F"

# Set date range (5 years of historical data)
end_date = datetime.now()
start_date = end_date - timedelta(days=5*365)

print(f"Fetching gold price data from {start_date.date()} to {end_date.date()}...")

# Download the data using yfinance
gold = yf.Ticker(gold_ticker)
data = gold.history(start=start_date, end=end_date)

# Display the first few rows to verify
print("\nFirst 5 rows of data:")
print(data.head())

# Save the data to a CSV file for later use
data.to_csv('gold_price_data.csv')
print("\n✅ Data saved to 'gold_price_data.csv'")

# Basic summary statistics
print("\nSummary Statistics:")
print(data['Close'].describe())

# Simple plot to visualise the data
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Close'], label='Gold Price (Close)')
plt.title('Gold Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.savefig('gold_price_trend.png')
plt.show()

print("\n📊 Price trend chart saved as 'gold_price_trend.png'")