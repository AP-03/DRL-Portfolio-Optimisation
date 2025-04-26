import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
import os

data_path = "./data"
csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]

dfs = {}

ticker_name_map = {
    'GSPC': 'S&P 500',
    'SP500-15': 'Materials',
    'SP500-20': 'Industrials',
    'SP500-25': 'Consumer Discretionary',
    'SP500-30': 'Consumer Staples',
    'SP500-35': 'Health Care',
    'SP500-40': 'Financials',
    'SP500-45': 'Information Technology',
    'SP500-50': 'Communication Services',
    'SP500-55': 'Utilities',
    'SP500-60': 'Real Estate',
    'GSPE': 'Energy'
}


# Pick a clean color palette
tickers_to_plot = [file.replace('.csv', '').replace('^', '') for file in csv_files if file.replace('.csv', '').replace('^', '') != 'VIX']
colors = cm.get_cmap('tab20', len(tickers_to_plot))

plt.figure(figsize=(15, 8))

for idx, file in enumerate(csv_files):
    ticker = file.replace('.csv', '').replace('^', '')

    if ticker == 'VIX':
        continue

    df = pd.read_csv(os.path.join(data_path, file))
    df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.dropna(subset=['Date'], inplace=True)

    adj_close_cols = [col for col in df.columns if 'Adj Close' in col or 'Adj' in col]

    if adj_close_cols:
        adj_close = pd.to_numeric(df[adj_close_cols[0]], errors='coerce')
        df = df.loc[adj_close.notna()].copy()
        adj_close = adj_close.loc[adj_close.notna()]

        cumulative_return = adj_close / adj_close.iloc[0]

        plt.plot(df['Date'], cumulative_return, label=ticker_name_map.get(ticker, ticker), color=colors(idx))

plt.title('S&P500 and 11 Subsectors: 2006 - 2021', fontsize=16)
plt.xlabel('Date')
plt.ylabel('Cumulative Return', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True)

plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
