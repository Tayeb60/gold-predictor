import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# 1. Load the data
df = pd.read_csv('gold_data_with_features.csv', index_col=0, parse_dates=True)

# 2. Select features and target
feature_columns = ['SMA_20', 'SMA_50', 'EMA_20', 'RSI_14', 'MACD', 
                   'MACD_Signal', 'MACD_Histogram', 'BB_Upper', 'BB_Lower', 'BB_Middle']
target_column = 'Close'

X = df[feature_columns].values
y = df[target_column].values

# 3. Split into train and test (chronological)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# 4. Train XGBoost model
model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

# 5. Make predictions
y_pred = model.predict(X_test)

# 6. Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n📊 Model Performance:")
print(f"Mean Absolute Error (MAE): ${mae:.2f}")
print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")

# 7. Directional accuracy
y_pred_flat = y_pred.flatten()
y_actual_flat = y_test.flatten()

y_pred_direction = np.where(y_pred_flat > np.roll(y_pred_flat, 1), 1, 0)[1:]
y_actual_direction = np.where(y_actual_flat > np.roll(y_actual_flat, 1), 1, 0)[1:]

directional_accuracy = np.mean(y_pred_direction == y_actual_direction)
print(f"Directional Accuracy: {directional_accuracy:.2%}")

# 8. Feature importance
feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n📊 Feature Importance (XGBoost):")
print(feature_importance)

# 9. Plot results
plt.figure(figsize=(14, 6))
plt.plot(df.index[-len(y_test):], y_test, label='Actual Price', color='gold')
plt.plot(df.index[-len(y_pred):], y_pred, label='Predicted Price', color='blue', linestyle='--')
plt.title('Gold Price Prediction with XGBoost')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.savefig('xgboost_gold_prediction.png', dpi=150)
plt.show()

# 10. Feature importance plot
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'], color='gold')
plt.xlabel('Importance')
plt.title('Feature Importance (XGBoost)')
plt.tight_layout()
plt.savefig('xgboost_feature_importance.png', dpi=150)
plt.show()

# 11. Save the model
model.save_model('xgboost_gold_model.json')
print("\n✅ Model saved as 'xgboost_gold_model.json'")