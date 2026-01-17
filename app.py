import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


from code.preprocessing import preprocess_data
from code.feature_engineering import create_features
from code.model_training import train_models
from code.evaluation import evaluate_model


st.set_page_config(
    page_title="Energy Consumption Forecasting",
    layout="wide"
)

# --------------------------------------------------
# Title
# --------------------------------------------------
st.title("🏠 Energy Consumption Forecasting in Smart Homes")
st.write(
    "This application predicts energy consumption using machine learning "
    "models based on historical and weather data."
)

# --------------------------------------------------
# File Upload
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Energy Consumption Dataset (CSV)",
    type=["csv"]
)

if uploaded_file is not None:
    # --------------------------------------------------
    # Load & preprocess data
    # --------------------------------------------------
    df = preprocess_data(uploaded_file)
    df = create_features(df)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # --------------------------------------------------
    # Train models
    # --------------------------------------------------
    models, X_test, y_test, imputer = train_models(df)

    # --------------------------------------------------
    # Model Selection
    # --------------------------------------------------
    st.subheader("🤖 Select Machine Learning Model")
    model_name = st.selectbox(
        "Choose Model",
        list(models.keys())
    )
    model = models[model_name]

    # --------------------------------------------------
    # Evaluation Metrics
    # --------------------------------------------------
    metrics = evaluate_model(model, X_test, y_test)

    mae = metrics["Model_MAE"]
    rmse = metrics["Model_RMSE"]
    r2 = metrics["Model_R2"]

    baseline_mae = metrics["Baseline_MAE"]
    baseline_rmse = metrics["Baseline_RMSE"]
    baseline_r2 = metrics["Baseline_R2"]

    st.subheader("📈 Model Evaluation Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{mae:.2f}")
    col2.metric("RMSE", f"{rmse:.2f}")
    col3.metric("R² Score", f"{r2:.2f}")

    # --------------------------------------------------
    # Actual vs Predicted Plot
    # --------------------------------------------------
    predictions = model.predict(X_test)

    st.subheader("📉 Actual vs Predicted Energy Consumption")
    fig1, ax1 = plt.subplots()
    ax1.plot(y_test.values[:100], label="Actual")
    ax1.plot(predictions[:100], label="Predicted")
    ax1.set_xlabel("Hourly Samples")
    ax1.set_ylabel("Energy Consumption (kWh)")
    ax1.legend()
    st.pyplot(fig1)

    # --------------------------------------------------
    # Energy Consumption Statistics
    # --------------------------------------------------
    st.subheader("📊 Energy Consumption Statistics")

    df_stats = df.copy()
    df_stats.set_index("Timestamp", inplace=True)

    avg_hourly = df_stats["Energy_Consumption"].mean()
    avg_daily = df_stats["Energy_Consumption"].resample("D").sum().mean()
    avg_weekly = df_stats["Energy_Consumption"].resample("W").sum().mean()
    avg_monthly = df_stats["Energy_Consumption"].resample("ME").sum().mean()

    st.write(f"**Average Hourly Consumption:** {avg_hourly:.2f} kWh")
    st.write(f"**Average Daily Consumption:** {avg_daily:.2f} kWh")
    st.write(f"**Average Weekly Consumption:** {avg_weekly:.2f} kWh")
    st.write(f"**Average Monthly Consumption:** {avg_monthly:.2f} kWh")

    # --------------------------------------------------
    # Future Energy Forecast (Next 24 Hours)
    # --------------------------------------------------
    st.subheader("🔮 Future Energy Consumption Forecast (Next 24 Hours)")

    def forecast_next_24_hours(df, model, imputer):
        last_row = df.iloc[-1].copy()
        forecasts = []

        for h in range(1, 25):
            row = last_row.copy()
            row["Hour"] = (row["Hour"] + h) % 24
            row["Lag_1"] = last_row["Energy_Consumption"]
            row["Lag_24"] = last_row["Energy_Consumption"]
            row["Rolling_Mean_24"] = last_row["Energy_Consumption"]

            X_future = row.drop(
                ["Timestamp", "Energy_Consumption"],
                errors="ignore"
            ).to_frame().T

            X_future = imputer.transform(X_future)
            pred = model.predict(X_future)[0]
            forecasts.append(pred)

        return forecasts

    future_forecast = forecast_next_24_hours(df, model, imputer)

    forecast_df = pd.DataFrame({
        "Hour Ahead": range(1, 25),
        "Predicted Energy (kWh)": future_forecast
    })

    st.dataframe(forecast_df)

    # --------------------------------------------------
    # Forecast Plot
    # --------------------------------------------------
    fig2, ax2 = plt.subplots()
    ax2.plot(future_forecast, marker="o")
    ax2.set_title("Future Energy Consumption Forecast")
    ax2.set_xlabel("Future Hours")
    ax2.set_ylabel("Energy Consumption (kWh)")
    ax2.grid(True)
    st.pyplot(fig2)

else:
    st.info("Please upload a CSV file to begin.")
