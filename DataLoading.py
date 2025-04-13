import pandas as pd
import yfinance as yf
import os

tickers = ['^GSPC', '^VIX', 'XLE', 'XLF', 'XLK', 'XLI', 
           'XLV', 'XLY', 'XLP', 'XLU', 'XLRE', 'XLB', 'XLC']

start_date = '2006-01-01'
end_date = '2021-12-31'

save_path = './data'  # folder to save csvs
os.makedirs(save_path, exist_ok=True)

for ticker in tickers:
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
    
    # Reset index to move 'Date' from index to a column
    df.reset_index(inplace=True)
    
    # Save to CSV with 'Date' as the first column
    df.to_csv(f'{save_path}/{ticker.replace("^", "")}.csv', index=False)
    
    print(f'Saved {ticker} to {save_path}/{ticker.replace("^", "")}.csv')
