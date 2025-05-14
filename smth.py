import pandas as pd
import yfinance as yf

ticker = "^SP500-60"
start_date = "2017-01-01"
end_date = "2017-12-31"

# Download the specific ticker for 2017
from curl_cffi import requests
session = requests.Session(impersonate="chrome")
df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False, session=session)

# Show shape and missing dates
print(f"Rows returned: {df.shape[0]}")  # Should be around 252 if complete

# Check what dates are missing
all_days = pd.date_range(start=start_date, end=end_date, freq="B")
missing_dates = all_days.difference(df.index)
print(f"Missing business days:\n{missing_dates}")