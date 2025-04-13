import pandas as pd
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

    print(f"\nHead of {file}:")
    print(df.head())

# Plot each ticker individually
for file, df in dfs.items():
    ticker = file.replace('.csv', '')

    adj_close_cols = [col for col in df.columns if 'Adj Close' in col or 'Adj' in col]

    if adj_close_cols:
        adj_close = pd.to_numeric(df[adj_close_cols[0]], errors='coerce')
        valid_mask = (adj_close > 0) & adj_close.notna()

        print(f'\nTicker: {ticker}')
        print(df.loc[valid_mask, ['Date', adj_close_cols[0]]].head())

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df.dropna(subset=['Date'], inplace=True)

        plt.figure(figsize=(12, 5))
        plt.plot(df.loc[valid_mask, 'Date'], adj_close[valid_mask])
        plt.title(f'{ticker} - Adj Close Price')
        plt.xlabel('Date')
        plt.ylabel('Adj Close Price')
        plt.grid(True)

        # Set x-axis major ticks to yearly
        plt.gca().xaxis.set_major_locator(mdates.YearLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
