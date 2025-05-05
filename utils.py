import pandas as pd


def build_env_windows(features_df, returns_df, start_year=2006, num_windows=10):
    windows = []

    for i in range(num_windows):
        # Compute dates
        train_start = pd.Timestamp(f"{start_year + i}-01-01")
        val_start   = pd.Timestamp(f"{start_year + i + 5}-01-01")
        test_start  = pd.Timestamp(f"{start_year + i + 6}-01-01")
        test_end    = pd.Timestamp(f"{start_year + i + 7}-01-01") - pd.Timedelta(days=1)

        # Build each split
        train_idx = (features_df.index >= train_start) & (features_df.index < val_start)
        val_idx   = (features_df.index >= val_start) & (features_df.index < test_start)
        test_idx  = (features_df.index >= test_start) & (features_df.index <= test_end)

        # Subset features and returns
        f_train, r_train = features_df.loc[train_idx], returns_df.loc[train_idx]
        f_val,   r_val   = features_df.loc[val_idx],   returns_df.loc[val_idx]
        f_test,  r_test  = features_df.loc[test_idx],  returns_df.loc[test_idx]

        # Append
        windows.append({
            "train": (f_train, r_train),
            "val":   (f_val, r_val),
            "test":  (f_test, r_test),
            "years": (train_start.year, test_end.year)
        })

    return windows



if __name__ == "__main__":

    features_df = pd.read_csv('./data/features_df.csv', index_col='Date', parse_dates=True)
    returns_df = pd.read_csv('./data/returns_df.csv', index_col='Date', parse_dates=True)
    
    env_windows = build_env_windows(features_df, returns_df)

    # For the first window
    first = env_windows[0]
    print(first["years"])       # (2006, 2012)
    print(len(first["train"][0]), len(first["val"][0]), len(first["test"][0]))
