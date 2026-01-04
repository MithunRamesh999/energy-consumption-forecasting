# code/preprocessing.py
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # Convert timestamp
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')

    # Forward fill missing values (Python 3.14 safe)
    df = df.ffill()

    # Extract time features
    df['Hour'] = df['Timestamp'].dt.hour
    df['Day'] = df['Timestamp'].dt.day
    df['Month'] = df['Timestamp'].dt.month
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek
    df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)

    # Normalize weather features
    scaler = MinMaxScaler()
    df[['Temperature', 'Humidity']] = scaler.fit_transform(
        df[['Temperature', 'Humidity']]
    )

    return df
