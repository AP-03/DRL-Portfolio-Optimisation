import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# === Load data ===
features = pd.read_csv('./data/features_df.csv', index_col='Date', parse_dates=True)
returns = pd.read_csv('./data/returns_df.csv', index_col='Date', parse_dates=True)

print("🔍 === Shape Check ===")
print(f"Features shape: {features.shape}")
print(f"Returns shape : {returns.shape}")
print("Same index     :", features.index.equals(returns.index))
print()

# === Check for NaNs ===
print("🔍 === NaN Check ===")
print("NaNs in features:")
print(features.isna().sum())
print("\nNaNs in returns:")
print(returns.isna().sum())
print()

# === Descriptive stats ===
print("🔍 === Feature Descriptives ===")
print(features[["vol20", "vol_ratio", "VIX"]].describe())
print()

# === Head/tail preview ===
print("🔍 === First/Last Rows ===")
print("\nFeatures head:")
print(features.head(3))
print("\nReturns head:")
print(returns.head(3))
print("\nFeatures tail:")
print(features.tail(3))
print("\nReturns tail:")
print(returns.tail(3))
print()

# === Plot vol indicators ===
features[["vol20", "vol_ratio", "VIX"]].plot(figsize=(12, 5), title="Volatility Features (Standardized)")
plt.grid(True)
plt.tight_layout()
plt.show()

# === Plot cumulative sector returns (subset) ===
returns.iloc[:, :].cumsum().plot(figsize=(12, 5), title="Cumulative Log Returns of 5 Sectors")
plt.grid(True)
plt.tight_layout()
plt.show()

# === Validate log return of one ticker ===
ticker = returns.columns[0]
price_path = f'./data/{ticker}.csv'

if os.path.exists(price_path):
    prices = pd.read_csv(price_path, parse_dates=['Date'], index_col='Date')["Adj Close"]
    prices = pd.to_numeric(prices, errors='coerce').dropna()
    manual_log_return = np.log(prices / prices.shift(1)).dropna()

    returns[ticker].plot(label="returns_df")
    manual_log_return.loc[returns.index].plot(label="manual", linestyle='--')
    plt.title(f"Validation: Log Returns of {ticker}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print(f"⚠️ Could not validate {ticker} — price file {price_path} not found.")

returns.stack().hist(bins=100, figsize=(10, 4), grid=True)
plt.title("Histogram of All Sector Log Returns")
plt.xlabel("Log Return")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
