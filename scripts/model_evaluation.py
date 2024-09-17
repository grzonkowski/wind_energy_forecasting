import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import lightgbm as lgb
import tensorflow as tf

# Load test data
X_test = np.load('prepared_data/X_test.npy')
y_test = np.load('prepared_data/y_test.npy')

# Ensure models and scalers directories exist
import os
os.makedirs('models', exist_ok=True)
os.makedirs('scalers', exist_ok=True)

# **SVR Model Evaluation**

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

# Evaluate the SVR model
mae_svr = mean_absolute_error(y_test, y_pred_svr)
rmse_svr = mean_squared_error(y_test, y_pred_svr, squared=False)
r2_svr = r2_score(y_test, y_pred_svr)

print(f'SVR Test MAE: {mae_svr}')
print(f'SVR Test RMSE: {rmse_svr}')
print(f'SVR Test R²: {r2_svr}')

# **LightGBM Model Evaluation**

# Load the trained LightGBM model
lgbm_model = lgb.Booster(model_file='models/lgbm_model.txt')

# Predict using the LightGBM model
y_pred_lgbm = lgbm_model.predict(X_test)

# Evaluate the LightGBM model
mae_lgbm = mean_absolute_error(y_test, y_pred_lgbm)
rmse_lgbm = mean_squared_error(y_test, y_pred_lgbm, squared=False)
r2_lgbm = r2_score(y_test, y_pred_lgbm)

print(f'LightGBM Test MAE: {mae_lgbm}')
print(f'LightGBM Test RMSE: {rmse_lgbm}')
print(f'LightGBM Test R²: {r2_lgbm}')

# **LSTM Model Evaluation**

# Reshape X_test for LSTM input
X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Load the trained LSTM model
lstm_model = tf.keras.models.load_model(
    'models/lstm_model.h5',
    custom_objects={'mse': tf.keras.losses.MeanSquaredError()},
    compile=True
)

# Predict using the LSTM model
y_pred_lstm = lstm_model.predict(X_test_lstm)

# Evaluate the LSTM model
mae_lstm = mean_absolute_error(y_test, y_pred_lstm)
rmse_lstm = mean_squared_error(y_test, y_pred_lstm, squared=False)
r2_lstm = r2_score(y_test, y_pred_lstm)

print(f'LSTM Test MAE: {mae_lstm}')
print(f'LSTM Test RMSE: {rmse_lstm}')
print(f'LSTM Test R²: {r2_lstm}')

# **CNN-LSTM Model Evaluation**

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
    custom_objects={'mae': tf.keras.losses.MeanAbsoluteError()},
    compile=True
)

# Predict using the CNN-LSTM model
y_pred_scaled = cnn_lstm_model.predict(X_test_seq)

# Inverse transform the predictions and actual values
y_pred_cnn_lstm = y_scaler.inverse_transform(y_pred_scaled)
y_test_seq = y_scaler.inverse_transform(y_test_seq)

# Evaluate the CNN-LSTM model
mae_cnn_lstm = mean_absolute_error(y_test_seq, y_pred_cnn_lstm)
rmse_cnn_lstm = mean_squared_error(y_test_seq, y_pred_cnn_lstm, squared=False)
r2_cnn_lstm = r2_score(y_test_seq, y_pred_cnn_lstm)

print(f'CNN-LSTM Test MAE: {mae_cnn_lstm}')
print(f'CNN-LSTM Test RMSE: {rmse_cnn_lstm}')
print(f'CNN-LSTM Test R²: {r2_cnn_lstm}')
