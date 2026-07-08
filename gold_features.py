import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data from the CSV file
df = pd.read_csv('gold_price_data.csv', index_col=0, parse_dates=True)

# 1. Moving Averages
df['SMA_20'] = df['Close'].rolling(window=20).mean()          # 20-day Simple Moving Average
df['SMA_50'] = df['Close'].rolling(window=50).mean()          # 50-day Simple Moving Average
df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()  # 20-day Exponential Moving Average

# 2. RSI (Relative Strength Index)
def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

df['RSI_14'] = calculate_rsi(df['Close'], 14)

# 3. MACD (Moving Average Convergence Divergence)
exp1 = df['Close'].ewm(span=12, adjust=False).mean()
exp2 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = exp1 - exp2
df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']

# 4. Bollinger Bands
df['BB_Middle'] = df['Close'].rolling(window=20).mean()
bb_std = df['Close'].rolling(window=20).std()
df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)

# 5. Price Change (Target Variable)
df['Price_Change'] = df['Close'].pct_change()          # Percentage change
df['Price_Change_Next'] = df['Price_Change'].shift(-1) # Tomorrow's change (for prediction)

# Remove rows with NaN values (first 50 days due to rolling calculations)
df_clean = df.dropna()

# Save the enriched data
df_clean.to_csv('gold_data_with_features.csv')
print(f"✅ Data with features saved to 'gold_data_with_features.csv'")
print(f"Total rows after cleaning: {len(df_clean)}")
print("\nFeature columns added:")
print(df_clean.columns.tolist())

# Plot some of the indicators
fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# Subplot 1: Price & Moving Averages
axes[0].plot(df_clean.index, df_clean['Close'], label='Close Price', color='gold', linewidth=2)
axes[0].plot(df_clean.index, df_clean['SMA_20'], label='SMA 20', linestyle='--')
axes[0].plot(df_clean.index, df_clean['SMA_50'], label='SMA 50', linestyle='--')
axes[0].set_title('Gold Price with Moving Averages')
axes[0].legend()
axes[0].grid(True)

# Subplot 2: RSI
axes[1].plot(df_clean.index, df_clean['RSI_14'], label='RSI 14', color='purple')
axes[1].axhline(y=70, color='red', linestyle='--', alpha=0.5)
axes[1].axhline(y=30, color='green', linestyle='--', alpha=0.5)
axes[1].set_title('RSI (Relative Strength Index)')
axes[1].legend()
axes[1].grid(True)

# Subplot 3: MACD
axes[2].plot(df_clean.index, df_clean['MACD'], label='MACD', color='blue')
axes[2].plot(df_clean.index, df_clean['MACD_Signal'], label='MACD Signal', color='red')
axes[2].bar(df_clean.index, df_clean['MACD_Histogram'], label='Histogram', color='green', alpha=0.3)
axes[2].set_title('MACD (Moving Average Convergence Divergence)')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.savefig('gold_features.png', dpi=150)
plt.show()