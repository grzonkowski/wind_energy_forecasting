import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
import joblib
import os
import lightgbm as lgb

# Ensure the results directory exists
os.makedirs('results', exist_ok=True)

# Define a function to calculate and plot residuals
def analyze_model_performance(model_name, y_true, y_pred, history=None):
    # Calculate evaluation metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)

    print(f"{model_name} Test MAE: {mae}")
    print(f"{model_name} Test RMSE: {rmse}")
    print(f"{model_name} Test R²: {r2}")

    # Residuals calculation
    residuals = y_true - y_pred

    # Plot residuals distribution
    plt.figure()
    sns.histplot(residuals, kde=True)
    plt.title(f'Residuals Distribution - {model_name} Model')
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.savefig(f'results/{model_name.lower()}_residuals.png')

    # If a history object exists (for neural networks), plot learning curves
    if history is not None:
        plt.figure()
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title(f'{model_name} Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'results/{model_name.lower()}_learning_curve.png')

# Load test data
X_test = np.load('prepared_data/X_test.npy')
y_test = np.load('prepared_data/y_test.npy')

# **1. SVR Model Analysis**

# Load the trained SVR model
svr_model = joblib.load('models/svr_model.pkl')

# Load scaler and PCA transformer used during training
scaler = joblib.load('models/scaler.pkl')
pca = joblib.load('models/pca.pkl')

# Apply the scaler and PCA to X_test
X_test_scaled = scaler.transform(X_test)
X_test_pca = pca.transform(X_test_scaled)

# Predict using the SVR model
y_pred_svr = svr_model.predict(X_test_pca)

# Analyze SVR model performance
analyze_model_performance('SVR', y_test, y_pred_svr)

# **2. LightGBM Model Analysis**

# Load the trained LightGBM model
lgbm_model = lgb.Booster(model_file='models/lgbm_model.txt')

# Predict using the LightGBM model
y_pred_lgbm = lgbm_model.predict(X_test)

# Analyze LightGBM model performance
analyze_model_performance('LightGBM', y_test, y_pred_lgbm)

# **3. LSTM Model Analysis**

# Load the trained LSTM model
lstm_model = tf.keras.models.load_model(
    'models/lstm_model.h5',
    custom_objects={'mse': tf.keras.losses.MeanSquaredError()},
    compile=True
)

# Reshape X_test for LSTM input
X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Predict using the LSTM model
y_pred_lstm = lstm_model.predict(X_test_lstm)

# Analyze LSTM model performance
analyze_model_performance('LSTM', y_test, y_pred_lstm.flatten())

# **4. CNN-LSTM Model Analysis**

# Load training history
history_cnn_lstm = joblib.load('models/cnn_lstm_history.pkl')

# Load scalers used during training
X_scaler = joblib.load('scalers/X_scaler.pkl')
y_scaler = joblib.load('scalers/y_scaler.pkl')

# Scale X_test and y_test
X_test_scaled = X_scaler.transform(X_test)
y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1))

# Create sequences for CNN-LSTM
sequence_length = 24

def create_sequences(X, y, seq_length):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])
    return np.array(X_seq), np.array(y_seq)

X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_scaled, sequence_length)

# Load the trained CNN-LSTM model
cnn_lstm_model = tf.keras.models.load_model(
    'models/cnn_lstm_model.h5',
    custom_objects={
        'mae': tf.keras.losses.MeanAbsoluteError(),
        'mse': tf.keras.losses.MeanSquaredError()
    },
    compile=True
)

# Predict using the CNN-LSTM model
y_pred_scaled = cnn_lstm_model.predict(X_test_seq)

# Inverse transform the predictions and actual values
y_pred_cnn_lstm = y_scaler.inverse_transform(y_pred_scaled)
y_test_seq = y_scaler.inverse_transform(y_test_seq)

# Analyze CNN-LSTM model performance
analyze_model_performance('CNN-LSTM', y_test_seq.flatten(), y_pred_cnn_lstm.flatten(), history_cnn_lstm)

print("All models' performance analysis completed!")
