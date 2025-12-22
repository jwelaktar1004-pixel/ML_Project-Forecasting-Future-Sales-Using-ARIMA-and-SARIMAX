import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Champagne Sales Forecast", layout="centered")

st.title("📈 Champagne Sales Forecasting Dashboard")
st.write("SARIMA-based Time Series Forecasting")

# Load dataset
df = pd.read_csv("perrin-freres-monthly-champagne.csv")
df.columns = ["Month", "Sales"]
df["Month"] = pd.to_datetime(df["Month"], errors="coerce")
df = df.dropna(subset=["Month"])
df.set_index("Month", inplace=True)

# Load trained model
with open("sarimax_model.pkl", "rb") as f:
    model = pickle.load(f)

# Forecast horizon slider
forecast_months = st.slider(
    "Select forecast horizon (months)",
    min_value=1,
    max_value=36,
    value=12
)

# Generate forecast
forecast = model.get_forecast(steps=forecast_months)
forecast_df = forecast.summary_frame()

# Plot
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(df.index, df["Sales"], label="Historical Sales", color="blue")
ax.plot(forecast_df.index, forecast_df["mean"], label="Forecast", color="orange")

ax.fill_between(
    forecast_df.index,
    forecast_df["mean_ci_lower"],
    forecast_df["mean_ci_upper"],
    color="orange",
    alpha=0.3,
    label="Confidence Interval"
)

ax.set_title("Sales Forecast with Confidence Interval")
ax.set_xlabel("Date")
ax.set_ylabel("Sales")
ax.legend()

st.pyplot(fig)

