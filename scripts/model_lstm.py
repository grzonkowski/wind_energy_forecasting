import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from datetime import datetime

# Start time
start_time = datetime.now()

print(f"[{start_time}] Starting LSTM Model Training...")

# Load data
X_train = np.load('prepared_data/X_train.npy')
y_train = np.load('prepared_data/y_train.npy')
X_val = np.load('prepared_data/X_val.npy')
y_val = np.load('prepared_data/y_val.npy')

# Check for NaNs
assert not np.any(np.isnan(X_train)), "X_train contains NaNs!"
assert not np.any(np.isnan(y_train)), "y_train contains NaNs!"
assert not np.any(np.isnan(X_val)), "X_val contains NaNs!"
assert not np.any(np.isnan(y_val)), "y_val contains NaNs!"

# Reshape for LSTM
X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_val_lstm = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))

# Model building with dropout and reduced learning rate
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]), return_sequences=True),
    tf.keras.layers.Dropout(0.2),  # Dropout for regularization
    tf.keras.layers.LSTM(50),
    tf.keras.layers.Dense(1)
])
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)  # Reduced learning rate
model.compile(optimizer=optimizer, loss='mse')

# Early stopping with reduced patience
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Training with early stopping
history = model.fit(X_train_lstm, y_train, epochs=100, batch_size=72, validation_data=(X_val_lstm, y_val),
                    verbose=2, shuffle=False, callbacks=[early_stopping])

# Saving the model
model.save('models/lstm_model.h5')
joblib.dump(history.history, 'models/lstm_history.pkl')

# Validation prediction
y_pred = model.predict(X_val_lstm)

# Check for NaNs in predictions
if np.any(np.isnan(y_pred)):
    print(f"[{datetime.now()}] Warning: y_pred contains NaN values. Handling them...")
    y_pred = np.nan_to_num(y_pred)

# Evaluation
mae = mean_absolute_error(y_val, y_pred)
rmse = mean_squared_error(y_val, y_pred, squared=False)
r2 = r2_score(y_val, y_pred)

# Output performance metrics
print(f"[{datetime.now()}] LSTM Model Training Finished")
print(f"Validation MAE: {mae}")
print(f"Validation RMSE: {rmse}")
print(f"Validation R²: {r2}")

# End time
end_time = datetime.now()
print(f"[{end_time}] Total Time for LSTM Model: {end_time - start_time}")
