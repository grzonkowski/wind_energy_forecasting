import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Constants
DATA_DIR = 'data/'
OUTPUT_DIR = 'prepared_data/'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data():
    # Load data from all turbine files
    all_data = []
    for filename in os.listdir(DATA_DIR):
        if filename.endswith('.csv'):
            filepath = os.path.join(DATA_DIR, filename)
            df = pd.read_csv(filepath, sep=';')
            all_data.append(df)
    data = pd.concat(all_data, ignore_index=True)
    return data


def handle_missing_data(data):
    # Continuous variables interpolation
    continuous_vars = ['Wind Speed (avg)', 'Active Power (avg)', 'Nacelle Position (avg)',
                       'T Outside Nacelle Level (avg)', 'T Generator 1 (avg)']
    # Impute NaNs using mean values for continuous variables
    imputer = SimpleImputer(strategy='mean')
    data[continuous_vars] = imputer.fit_transform(data[continuous_vars])

    return data


def detect_outliers(data):
    from scipy import stats
    continuous_vars = ['Wind Speed (avg)', 'Active Power (avg)']
    z_scores = np.abs(stats.zscore(data[continuous_vars]))
    data = data[(z_scores < 3).all(axis=1)]
    return data


def feature_engineering(data):
    # Operational Status
    data['Operational Status'] = np.where(
        (data['Wind Speed (avg)'] >= 3.5) & (data['Active Power (avg)'] <= 50), 0,
        np.where(
            (data['Wind Speed (avg)'] >= 3.5) & (data['Active Power (avg)'] > 50), 1, 2
        )
    )
    # Lag Features
    data.sort_values(['Date (Plant)', 'Time (Plant)'], inplace=True)
    data['Wind Speed Lag1'] = data['Wind Speed (avg)'].shift(1)
    data['Active Power Lag1'] = data['Active Power (avg)'].shift(1)
    # Rolling Statistics
    data['Wind Speed MA3'] = data['Wind Speed (avg)'].rolling(window=3).mean()
    data['Active Power MA3'] = data['Active Power (avg)'].rolling(window=3).mean()
    # Cyclic Encoding
    data['Datetime'] = pd.to_datetime(data['Date (Plant)'] + ' ' + data['Time (Plant)'], format='%d/%m/%Y %H:%M:%S')
    data['Hour'] = data['Datetime'].dt.hour
    data['DayOfWeek'] = data['Datetime'].dt.dayofweek
    data['Month'] = data['Datetime'].dt.month
    data['Hour_sin'] = np.sin(2 * np.pi * data['Hour'] / 24)
    data['Hour_cos'] = np.cos(2 * np.pi * data['Hour'] / 24)
    data['DayOfWeek_sin'] = np.sin(2 * np.pi * data['DayOfWeek'] / 7)
    data['DayOfWeek_cos'] = np.cos(2 * np.pi * data['DayOfWeek'] / 7)
    data['Month_sin'] = np.sin(2 * np.pi * data['Month'] / 12)
    data['Month_cos'] = np.cos(2 * np.pi * data['Month'] / 12)
    # One-Hot Encoding
    data = pd.get_dummies(data, columns=['Operational Status'], prefix='OpStatus')
    return data


def split_and_scale_data(data):
    # Ensure 'Datetime' column exists before dropping other columns
    assert 'Datetime' in data.columns, "'Datetime' column is missing!"

    # Features and target
    target = 'Active Power (avg)'

    # Drop non-numerical and non-relevant columns, including 'Datetime' for scaling
    features = data.drop(columns=['Identifier (Plant)', 'Date (Plant)', 'Time (Plant)', 'Datetime', target])

    # Time-based splitting
    data['Year'] = data['Datetime'].dt.year
    train_data = data[data['Year'] <= 2021]
    val_data = data[data['Year'] == 2022]
    test_data = data[data['Year'] == 2023]

    X_train = train_data[features.columns]
    y_train = train_data[target]
    X_val = val_data[features.columns]
    y_val = val_data[target]
    X_test = test_data[features.columns]
    y_test = test_data[target]

    # Impute NaNs before scaling (if any remaining)
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)
    X_test = imputer.transform(X_test)

    # Scaling
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test


def save_prepared_data(X_train, y_train, X_val, y_val, X_test, y_test):
    np.save(os.path.join(OUTPUT_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(OUTPUT_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(OUTPUT_DIR, 'X_val.npy'), X_val)
    np.save(os.path.join(OUTPUT_DIR, 'y_val.npy'), y_val)
    np.save(os.path.join(OUTPUT_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(OUTPUT_DIR, 'y_test.npy'), y_test)


def main():
    data = load_data()
    data = handle_missing_data(data)
    data = detect_outliers(data)
    data = feature_engineering(data)
    X_train, y_train, X_val, y_val, X_test, y_test = split_and_scale_data(data)
    save_prepared_data(X_train, y_train, X_val, y_val, X_test, y_test)


if __name__ == '__main__':
    main()
