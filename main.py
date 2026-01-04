import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from code.preprocessing import preprocess_data
from code.feature_engineering import create_features
from code.model_training import train_models
from code.evaluation import evaluate_model

# --------------------------------------------------
# 1. Load & preprocess data
# --------------------------------------------------
df = preprocess_data("dataset/energy_data.csv")
print("Rows after preprocessing:", len(df))

# --------------------------------------------------
# 2. Feature engineering
# --------------------------------------------------
df = create_features(df)
print("Rows after feature engineering:", len(df))

# --------------------------------------------------
# 3. Train models
# --------------------------------------------------
models, X_test, y_test, imputer = train_models(df)

# --------------------------------------------------
# 4. Evaluate models
# --------------------------------------------------
print("\nModel Performance Results:")
for name, model in models.items():
    mae, rmse, r2 = evaluate_model(model, X_test, y_test)
    print(f"\n{name}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")

# --------------------------------------------------
# 5. Energy consumption statistics
# --------------------------------------------------
print("\n📊 Energy Consumption Statistics")

df_stats = df.copy()
df_stats.set_index('Timestamp', inplace=True)

avg_hourly = df_stats['Energy_Consumption'].mean()
daily = df_stats['Energy_Consumption'].resample('D').sum().mean()
weekly = df_stats['Energy_Consumption'].resample('W').sum().mean()
monthly = df_stats['Energy_Consumption'].resample('ME').sum().mean()

print(f"Average Hourly: {avg_hourly:.2f} kWh")
print(f"Average Daily: {daily:.2f} kWh")
print(f"Average Weekly: {weekly:.2f} kWh")
print(f"Average Monthly: {monthly:.2f} kWh")

# --------------------------------------------------
# 6. Historical prediction (Method 1)
# --------------------------------------------------
best_model = models["XGBoost"]
historical_predictions = best_model.predict(X_test)

print("\nSample Historical Predictions:")
for i in range(5):
    print(f"Hour {i+1}: {historical_predictions[i]:.2f} kWh")

# --------------------------------------------------
# 7. Future forecasting (Next 24 hours)
# --------------------------------------------------
def forecast_next_24_hours(df, model, imputer):
    last_row = df.iloc[-1].copy()
    future_predictions = []

    for h in range(1, 25):
        row = last_row.copy()
        row['Hour'] = (row['Hour'] + h) % 24
        row['Lag_1'] = last_row['Energy_Consumption']
        row['Lag_24'] = last_row['Energy_Consumption']
        row['Rolling_Mean_24'] = last_row['Energy_Consumption']

        X_future = row.drop(
            ['Timestamp', 'Energy_Consumption'],
            errors='ignore'
        ).to_frame().T

        X_future = imputer.transform(X_future)
        pred = model.predict(X_future)[0]
        future_predictions.append(pred)

    return future_predictions

future_forecast = forecast_next_24_hours(df, best_model, imputer)

print("\n🔮 Future Energy Consumption Forecast (Next 24 Hours):")
for i, val in enumerate(future_forecast, 1):
    print(f"Hour +{i}: {val:.2f} kWh")

# --------------------------------------------------
# 8. Plot historical vs predicted
# --------------------------------------------------
plt.figure(figsize=(8, 4))
plt.plot(y_test.values[:100], label="Actual")
plt.plot(historical_predictions[:100], label="Predicted")
plt.title("Actual vs Predicted Energy Consumption")
plt.xlabel("Hourly Samples")
plt.ylabel("Energy Consumption (kWh)")
plt.legend()
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 9. Plot future forecast
# --------------------------------------------------
plt.figure(figsize=(8, 4))
plt.plot(future_forecast, marker='o')
plt.title("Future Energy Consumption Forecast (Next 24 Hours)")
plt.xlabel("Future Hours")
plt.ylabel("Energy Consumption (kWh)")
plt.grid(True)
plt.tight_layout()
plt.show()
