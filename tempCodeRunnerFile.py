import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

data_path = "./data"
csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]

dfs = {}

# Load data
for file in csv_files:
    file_path = os.path.join(data_path, file)
    df = pd.read_csv(file_path)

    dfs[file] = df

plt.figure(figsize=(15, 8))

for file, df in dfs.items():
    ticker = file.replace('.csv', '')

    if ticker == 'VIX':  # Skip VIX
        continue

    adj_close_cols = [col for col in df.columns if 'Adj Close' in col or 'Adj' in col]

    if adj_close_cols:
        adj_close = pd.to_numeric(df[adj_close_cols[0]], errors='coerce')
        valid_mask = adj_close.notna()

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.dropna(subset=['Date'], inplace=True)

        log_returns = (adj_close / adj_close.shift(1)).apply(np.log)
        cumulative_log_return = log_returns.cumsum().apply(np.exp)

        plt.plot(df.loc[valid_mask, 'Date'], cumulative_log_return[valid_mask], label=ticker)

plt.title('S&P500 and 11 Subsectors: 2006 - 2021')
plt.xlabel('Date')
plt.ylabel('Cumulative Return (Log Returns)')
plt.legend()
plt.grid(True)

plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

