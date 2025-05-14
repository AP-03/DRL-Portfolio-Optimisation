import os
import pandas as pd
import numpy as np

# ————————————————————————————————————————
# 1.  File locations
# ————————————————————————————————————————
DATA_DIR = "./data"

SECTOR_FILES = [
    "GSPE.csv",          # defensive “cash” proxy in the JPM study
    "SP500-15.csv", "SP500-20.csv", "SP500-25.csv", "SP500-30.csv",
    "SP500-35.csv", "SP500-40.csv", "SP500-45.csv",
    "SP500-50.csv", "SP500-55.csv", "SP500-60.csv"
]

SP500_FILE = "GSPC.csv"          # S&P-500 composite
VIX_FILE   = "VIX.csv"           # CBOE volatility index

# ————————————————————————————————————————
# 2.  Load sector prices  ➜  daily log returns
# ————————————————————————————————————————
price_frames = {}
for fname in SECTOR_FILES:
    ticker = fname.replace(".csv", "")
    df = pd.read_csv(os.path.join(DATA_DIR, fname), parse_dates=["Date"])
    df = df.set_index("Date")

    # prefer 'Adj Close' if present, else first numeric column
    col = "Adj Close" if "Adj Close" in df.columns else df.columns[0]
    series = pd.to_numeric(df[col], errors="coerce")

    price_frames[ticker] = series

# 🛠️ Insert full business day index and forward-fill missing dates
all_dates = pd.date_range(start='2005-09-01', end='2021-12-31', freq='B')  # Business days
for ticker in price_frames:
    price_frames[ticker] = price_frames[ticker].reindex(all_dates).ffill()

# Combine into single DataFrame
adj_close_df = pd.concat(price_frames.values(), axis=1)
adj_close_df.columns = price_frames.keys()

# drop rows only when **all** sector prices are NA  (e.g., holiday)
adj_close_df = adj_close_df.dropna(how="all").sort_index()


# daily log-returns
log_returns = np.log(adj_close_df / adj_close_df.shift(1)).dropna()
returns_df = log_returns.copy()

price_df = adj_close_df.loc[log_returns.index].copy()
# Don't rename columns — we keep them as original asset names



# ————————————————————————————————————————
# 3.  Market-regime features (SP-500 & VIX)
# ————————————————————————————————————————
sp500 = pd.read_csv(os.path.join(DATA_DIR, SP500_FILE),
                    parse_dates=["Date"], index_col="Date")
sp500_adj = pd.to_numeric(sp500["Adj Close"], errors="coerce").ffill()
sp500_ret = np.log(sp500_adj / sp500_adj.shift(1)).dropna()

vol20 = sp500_ret.rolling(20).std()
vol60 = sp500_ret.rolling(60).std()
vol_ratio = vol20 / vol60

vix = pd.read_csv(os.path.join(DATA_DIR, VIX_FILE),
                  parse_dates=["Date"], index_col="Date")
vix_adj = pd.to_numeric(vix["Adj Close"], errors="coerce").ffill()

# ————————————————————————————————————————
# 4.  Assemble features & leakage-safe standardisation
# ————————————————————————————————————————
features_df = pd.DataFrame(index=log_returns.index)
features_df["SP500_logret"] = sp500_ret
features_df["vol20"]        = vol20
features_df["vol_ratio"]    = vol_ratio
features_df["VIX"]          = vix_adj

# expanding-window z-score on regime features (paper §4)  
for col in ["SP500_logret", "vol20", "vol_ratio", "VIX"]:
    exp_mean = features_df[col].expanding().mean()
    exp_std  = features_df[col].expanding().std()
    features_df[col] = (features_df[col] - exp_mean) / exp_std

# drop first 60 days (where vol60 is NaN) + align returns
features_df = features_df.merge(price_df, left_index=True, right_index=True, how='left')
features_df = features_df.dropna()
returns_df  = returns_df.loc[features_df.index]

# Example: trim everything before January 1, 2006
EXPERIMENT_START_DATE = "2006-01-01"

features_df.index.name = "Date"
returns_df.index.name = "Date"


# ————————————————————————————————————————
# 5.  Save to disk
# ————————————————————————————————————————
os.makedirs(DATA_DIR, exist_ok=True)
features_df.to_csv(os.path.join(DATA_DIR, "features_df.csv"))
returns_df.to_csv(os.path.join(DATA_DIR, "returns_df.csv"))

print("✅ Preprocessing complete — features_df.csv and returns_df.csv written.")
