import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Champagne Sales Forecast", layout="centered")

# ---------------- TITLE ----------------
st.title("📈 Champagne Sales Forecasting Dashboard")
st.write("SARIMA-based Time Series Forecasting")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("perrin-freres-monthly-champagne.csv")
df.columns = ["Month", "Sales"]
df["Month"] = pd.to_datetime(df["Month"], errors="coerce")
df = df.dropna(subset=["Month"])
df.set_index("Month", inplace=True)

# ---------------- LOAD MODEL ----------------
with open("sarimax_model.pkl", "rb") as f:
    model = pickle.load(f)

# ---------------- USER INPUT ----------------
forecast_months = st.slider(
    "Select forecast horizon (months)",
    min_value=1,
    max_value=36,
    value=12
)

# ---------------- MODEL INFO PANEL (DYNAMIC) ----------------
st.markdown("### 🧠 Model Information")
st.info(
    f"""
    **Model:** SARIMAX  
    **Seasonality:** 12 months  
    **Training Period:** 1964 – 1972  
    **Selected Forecast Horizon:** {forecast_months} months  
    **Forecast Output:** Monthly sales with 95% confidence interval
    """
)

# ---------------- FORECAST ----------------
forecast = model.get_forecast(steps=forecast_months)
forecast_df = forecast.summary_frame()

# ---------------- BUSINESS METRICS (DYNAMIC) ----------------
last_actual = df["Sales"].iloc[-1]
selected_forecast = forecast_df["mean"].iloc[-1]
growth_pct = ((selected_forecast - last_actual) / last_actual) * 100

st.markdown("### 📊 Key Forecast Metrics")
col1, col2, col3 = st.columns(3)

col1.metric("Last Observed Sales", f"{last_actual:,.0f}")
col2.metric(
    f"Forecast after {forecast_months} months",
    f"{selected_forecast:,.0f}"
)
col3.metric("Expected Change", f"{growth_pct:.2f}%")

# ---------------- PLOT ----------------
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

# ---------------- CONFIDENCE INTERPRETATION ----------------
st.caption(
    "🟠 The shaded area represents the **95% confidence interval**, indicating the range "
    "within which future sales values are most likely to fall."
)

# ---------------- DOWNLOAD FORECAST ----------------
st.markdown("### 📥 Download Forecast Data")

download_df = forecast_df.reset_index()
download_df.rename(columns={"index": "Date"}, inplace=True)

st.download_button(
    label="Download forecast as CSV",
    data=download_df.to_csv(index=False),
    file_name="champagne_sales_forecast.csv",
    mime="text/csv"
)

# ---------------- FOOTER ----------------
st.caption("Built by Jwel Aktar | SARIMAX Time Series Forecasting | Streamlit Cloud")
