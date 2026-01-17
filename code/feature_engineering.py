# code/feature_engineering.py

def create_features(df):

    if len(df) < 25:
        raise ValueError(
            "Dataset too small. Minimum 25 rows required."
        )

    # Lag features (will create NaN – expected)
    df['Lag_1'] = df['Energy_Consumption'].shift(1)
    df['Lag_24'] = df['Energy_Consumption'].shift(24)

    # Rolling mean (safe)
    df['Rolling_Mean_24'] = (
        df['Energy_Consumption']
        .rolling(window=24, min_periods=1)
        .mean()
    )
    df = df.dropna().reset_index(drop=True)
    return df
