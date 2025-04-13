import pandas as pd
import numpy as np

class JPMDataPreprocessor:
    def __init__(self, data):
        """
        data: dict of {ticker: pd.DataFrame with ['Date', 'Adj Close']}
        """
        self.data = {k: v.set_index('Date').sort_index() for k, v in data.items()}
        self.tickers = list(data.keys())
        self.asset_tickers = [ticker for ticker in self.tickers if ticker not in ['VIX']]

    @staticmethod
    def log_returns(prices):
        return np.log(prices / prices.shift(1))

    @staticmethod
    def expanding_standardize(series):
        expanding_mean = series.expanding().mean()
        expanding_std = series.expanding().std().replace(0, np.nan)
        return (series - expanding_mean) / expanding_std

    def compute_features(self):
        # Compute log returns for assets (except VIX)
        returns = {ticker: self.log_returns(df['Adj Close']) for ticker, df in self.data.items() if ticker != 'VIX'}
        
        # Compute vol20, vol60, vol_ratio from SP500 returns
        spx_returns = returns['SP500']
        vol20 = spx_returns.rolling(window=20).std()
        vol60 = spx_returns.rolling(window=60).std()
        vol_ratio = vol20 / vol60

        # Standardize vol20, vol_ratio, and VIX using expanding window
        vol20_std = self.expanding_standardize(vol20)
        vol_ratio_std = self.expanding_standardize(vol_ratio)
        vix_std = self.expanding_standardize(self.data['VIX']['Adj Close'])

        # Combine features
        features = pd.DataFrame({
            'vol20_std': vol20_std,
            'vol_ratio_std': vol_ratio_std,
            'vix_std': vix_std
        })

        # Store
        self.returns = returns
        self.features = features
        
        return returns, features


# Example Usage:
# data = {
#     'SP500': pd.read_csv('SP500.csv'),
#     'VIX': pd.read_csv('VIX.csv'),
#     'XLF': pd.read_csv('XLF.csv'),
#     ... other ETFs
# }

# preprocessor = JPMDataPreprocessor(data)
# returns, features = preprocessor.compute_features()

# returns -> dictionary of log returns per asset
# features -> DataFrame with vol20_std, vol_ratio_std, vix_std
