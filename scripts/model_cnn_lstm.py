import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import joblib
from datetime import datetime
import os

# Start time
start_time = datetime.now()
print(f"[{start_time}] Starting CNN-LSTM Model Training...")

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('scalers', exist_ok=True)

# Load data
X_train = np.load('prepared_data/X_train.npy').astype(np.float32)
y_train = np.load('prepared_data/y_train.npy').astype(np.float32)
X_val = np.load('prepared_data/X_val.npy').astype(np.float32)
y_val = np.load('prepared_data/y_val.npy').astype(np.float32)

# Check for NaNs
assert not np.any(np.isnan(X_train)), "X_train contains NaNs!"
assert not np.any(np.isnan(y_train)), "y_train contains NaNs!"
assert not np.any(np.isnan(X_val)), "X_val contains NaNs!"
assert not np.any(np.isnan(y_val)), "y_val contains NaNs!"

# Feature Scaling
X_scaler = MinMaxScaler()
X_train_scaled = X_scaler.fit_transform(X_train)
X_val_scaled = X_scaler.transform(X_val)
joblib.dump(X_scaler, 'scalers/X_scaler.pkl')

# Scaling target variable
y_scaler = MinMaxScaler()
y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1))
y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1))
joblib.dump(y_scaler, 'scalers/y_scaler.pkl')

# Increase sequence length
sequence_length = 24
n_features = X_train_scaled.shape[1]

# Create sequences with overlap
def create_sequences(X, y, seq_length):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        y_seq.append(y[i+seq_length])
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)

# To manage memory, process in chunks if necessary
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_scaled, sequence_length)
X_val_seq, y_val_seq = create_sequences(X_val_scaled, y_val_scaled, sequence_length)

# Define model
model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=(sequence_length, n_features)),
    tf.keras.layers.Conv1D(filters=64, kernel_size=5, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100, activation='relu', return_sequences=True)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(50, activation='relu'),
    tf.keras.layers.Dense(50, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dense(1)
])

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer, loss='mae', metrics=['mse'])

# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
callbacks = [early_stopping, lr_scheduler]

# Training
history = model.fit(
    X_train_seq, y_train_seq,
    epochs=100,
    batch_size=64,
    validation_data=(X_val_seq, y_val_seq),
    verbose=2,
    shuffle=True,
    callbacks=callbacks
)

# Save model and history
model.save('models/cnn_lstm_model.h5')
joblib.dump(history.history, 'models/cnn_lstm_history.pkl')

# Predict and inverse transform
y_pred_scaled = model.predict(X_val_seq)
y_pred = y_scaler.inverse_transform(y_pred_scaled)
y_true = y_scaler.inverse_transform(y_val_seq)

# Evaluation Metrics
mae = mean_absolute_error(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred, squared=False)
r2 = r2_score(y_true, y_pred)

print(f"[{datetime.now()}] CNN-LSTM Model Training Finished")
print(f"Validation MAE: {mae}")
print(f"Validation RMSE: {rmse}")
print(f"Validation R²: {r2}")

# End time
end_time = datetime.now()
print(f"[{end_time}] Total Time for CNN-LSTM Model: {end_time - start_time}")
