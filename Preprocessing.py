import pandas as pd
import numpy as np
import os

# Define data path
data_path = "./data"

# List all files you want to include
sector_files = [
    "GSPE.csv", "SP500-15.csv", "SP500-20.csv", "SP500-25.csv", "SP500-30.csv",
    "SP500-35.csv", "SP500-40.csv", "SP500-45.csv", "SP500-50.csv", "SP500-55.csv", "SP500-60.csv"
]

sp500_file = "GSPC.csv"
vix_file = "VIX.csv"

# Load and process sector adj close prices
price_data = {}

for file in sector_files:
    ticker = file.replace('.csv', '')
    df = pd.read_csv(os.path.join(data_path, file), parse_dates=["Date"])
    df = df.set_index("Date")
    colname = df.columns[0]
    price_data[ticker] = pd.to_numeric(df[colname], errors='coerce')

# Combine into one DataFrame
adj_close_df = pd.concat(price_data.values(), axis=1)
adj_close_df.columns = price_data.keys()
adj_close_df = adj_close_df.sort_index().dropna(how="any")  # drop rows with any NaN

# Compute daily log returns
log_returns = np.log(adj_close_df / adj_close_df.shift(1)).dropna()

# Save returns_df (target for reward computation)
returns_df = log_returns.copy()
returns_df.to_csv("./data/returns_df.csv")

# Load S&P500 and compute volatility indicators
sp500 = pd.read_csv(os.path.join(data_path, sp500_file), parse_dates=["Date"], index_col="Date")
sp500_adj = pd.to_numeric(sp500["Adj Close"], errors='coerce').dropna()
sp500_ret = np.log(sp500_adj / sp500_adj.shift(1)).dropna()

vol20 = sp500_ret.rolling(window=20).std()
vol60 = sp500_ret.rolling(window=60).std()
vol_ratio = vol20 / vol60

# Load VIX
vix = pd.read_csv(os.path.join(data_path, vix_file), parse_dates=["Date"], index_col="Date")
vix_adj = pd.to_numeric(vix["Adj Close"], errors='coerce')

# Align everything
features_df = log_returns.copy()
features_df["SP500_logret"] = sp500_ret
features_df["vol20"] = vol20
features_df["vol_ratio"] = vol_ratio
features_df["VIX"] = vix_adj

# Standardize selected features (but only using past data to avoid leakage)
for col in ["SP500_logret", "vol20", "vol_ratio", "VIX"]:
    expanding_mean = features_df[col].expanding().mean()
    expanding_std = features_df[col].expanding().std()
    features_df[col] = (features_df[col] - expanding_mean) / expanding_std

features_df = features_df.dropna()
returns_df = returns_df.loc[features_df.index]

# Save output
features_df.to_csv("./data/features_df.csv")
returns_df.to_csv("./data/returns_df.csv")

print("✅ Done: features_df and returns_df saved to ./data/")
