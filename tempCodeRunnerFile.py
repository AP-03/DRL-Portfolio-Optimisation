import pandas as pd
import matplotlib.pyplot as plt
import os

# Path to your data directory
data_path = "./data"

# List of all CSV files
csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]

dfs = {}

# Load and print head of each file
for file in csv_files:
    file_path = os.path.join(data_path, file)
    df = pd.read_csv(file_path)
    
    # Store in dictionary for later plotting
    dfs[file] = df
    
    print(f"\nHead of {file}:")
    print(df.head())

# Plotting all Adj Close values
plt.figure(figsize=(15, 8))

for file, df in dfs.items():
    ticker = file.replace('.csv', '')
    
    # Handle different formats of 'Adj Close' column name
    adj_close_cols = [col for col in df.columns if 'Adj Close' in col or 'Adj Close' in col or 'Adj' in col]
    
    if adj_close_cols:
        plt.plot(pd.to_datetime(df['Date']), df[adj_close_cols[0]], label=ticker)

plt.title('Adj Close Price Comparison')
plt.xlabel('Date')
plt.ylabel('Adj Close Price')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
