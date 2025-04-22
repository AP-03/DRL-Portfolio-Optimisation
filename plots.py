import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# Color map per sector (customized or from matplotlib colormap)
fixed_color_map = {
    'GSPC': 'red',
    'SP500-15': 'green',                   # Materials
    'SP500-20': 'violet',                  # Industrials
    'SP500-25': 'deepskyblue',             # Consumer Discretionary
    'SP500-30': 'limegreen',               # Consumer Staples
    'SP500-35': 'orange',                  # Health Care
    'SP500-40': 'pink',                    # Financials
    'SP500-45': 'brown',                   # Info Tech
    'SP500-50': 'cyan',                    # Comm Services
    'SP500-55': 'mediumpurple',            # Utilities
    'SP500-60': 'gray',                    # Real Estate
    'GSPE': 'darkgoldenrod'               # Energy (approx for SP500-10)
}

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

# Set up
data_path = "./data"
csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]

plt.figure(figsize=(15, 8))

# Plot loop
for file in csv_files:
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

        cumulative_return = adj_close / adj_close.iloc[0]
        plt.plot(
            df['Date'],
            cumulative_return,
            label=ticker_name_map.get(ticker, ticker),
            color=fixed_color_map.get(ticker, 'black')  # default to black if not found
        )

# Final touches
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
