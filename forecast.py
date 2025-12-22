import pandas as pd
import pickle
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Load dataset
df = pd.read_csv("perrin-freres-monthly-champagne.csv")

# Rename columns
df.columns = ["Month", "Sales"]

# Safe datetime conversion
df["Month"] = pd.to_datetime(df["Month"], errors="coerce")

# Remove invalid rows
df = df.dropna(subset=["Month"])

# Set index
df.set_index("Month", inplace=True)

# Train SARIMAX model
model = SARIMAX(
    df["Sales"],
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12)
)

results = model.fit()

# Save model
with open("sarimax_model.pkl", "wb") as f:
    pickle.dump(results, f)

print("Model saved as sarimax_model.pkl")
