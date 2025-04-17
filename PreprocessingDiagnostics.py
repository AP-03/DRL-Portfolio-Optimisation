import pandas as pd
import os

data_path = "./data"
csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]

print("=== DATASET DIAGNOSTICS ===\n")

for file in csv_files:
    file_path = os.path.join(data_path, file)
    df = pd.read_csv(file_path)

    adj_close = df["Adj Close"]

    Date=df['Date']

    Date = pd.to_datetime(Date, errors='coerce')

    # Check and print diagnostics
    print(f"--- {file} ---")
    print(f"Shape: {df.shape}")
    print(f"NaNs in Date: {Date.isna().sum()}")

    print(f"NaNs in 'adj_close': {adj_close.isna().sum()}")

    
    print()

    print("=== LOCATING NaNs ===\n")

for file in csv_files:
    file_path = os.path.join(data_path, file)
    df = pd.read_csv(file_path)

    # Rename date column if unnamed
    if 'Date' not in df.columns:
        df.rename(columns={df.columns[0]: 'Date'}, inplace=True)

    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Detect adj close column
    adj_close_col = next((col for col in df.columns if 'Adj Close' in col or 'Adj' in col), None)

    if adj_close_col:
        df[adj_close_col] = pd.to_numeric(df[adj_close_col], errors='coerce')

        nan_rows = df[df['Date'].isna() | df[adj_close_col].isna()]
        if not nan_rows.empty:
            print(f"\n--- {file} ---")
            print(nan_rows)
    else:
        print(f"\n⚠️ {file} has no 'Adj Close' column.\n")
