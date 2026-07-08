import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

# 1. Load the data
df = pd.read_csv('gold_data_with_features.csv', index_col=0, parse_dates=True)

# 2. Select features and target
feature_columns = ['SMA_20', 'SMA_50', 'EMA_20', 'RSI_14', 'MACD', 
                   'MACD_Signal', 'MACD_Histogram', 'BB_Upper', 'BB_Lower', 'BB_Middle']
target_column = 'Close'

# 3. Scale the data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(df[feature_columns].values)
y_scaled = scaler_y.fit_transform(df[[target_column]].values)

# 4. Create sequences (20 time steps for speed)
def create_sequences(X, y, time_steps=20):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i+time_steps])
        ys.append(y[i+time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 20
X_seq, y_seq = create_sequences(X_scaled, y_scaled, time_steps)

# 5. Split into train and test
split = int(0.8 * len(X_seq))
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Input shape: {X_train.shape[1]} time steps, {X_train.shape[2]} features")

# 6. Build LSTM model (smaller = faster)
model = Sequential([
    LSTM(units=32, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(units=32, return_sequences=False),
    Dropout(0.2),
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# 7. Train (fewer epochs)
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# 8. Predict and evaluate
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_actual = scaler_y.inverse_transform(y_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(y_actual, y_pred)
mse = mean_squared_error(y_actual, y_pred)
rmse = np.sqrt(mse)

print(f"\n📊 Model Performance:")
print(f"Mean Absolute Error (MAE): ${mae:.2f}")
print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")

# Directional accuracy
y_pred_flat = y_pred.flatten()
y_actual_flat = y_actual.flatten()

y_pred_direction = np.where(y_pred_flat > np.roll(y_pred_flat, 1), 1, 0)[1:]
y_actual_direction = np.where(y_actual_flat > np.roll(y_actual_flat, 1), 1, 0)[1:]

directional_accuracy = np.mean(y_pred_direction == y_actual_direction)
print(f"Directional Accuracy: {directional_accuracy:.2%}")

# Plot results
plt.figure(figsize=(14, 6))
plt.plot(df.index[-len(y_actual):], y_actual, label='Actual Price', color='gold')
plt.plot(df.index[-len(y_pred):], y_pred, label='Predicted Price', color='blue', linestyle='--')
plt.title('Gold Price Prediction with LSTM (Fast Version)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.savefig('lstm_gold_fast.png', dpi=150)
plt.show()

# Plot training loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('lstm_fast_loss.png', dpi=150)
plt.show()

# Save the model
model.save('gold_lstm_model.h5')
print("\n✅ Model saved as 'gold_lstm_model.h5'")