import numpy as np
from sklearn.svm import SVR, LinearSVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

# Load data
X_train = np.load('prepared_data/X_train.npy')
y_train = np.load('prepared_data/y_train.npy')
X_val = np.load('prepared_data/X_val.npy')
y_val = np.load('prepared_data/y_val.npy')

## Impute missing values (fill missing values with the mean)
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)
X_val = imputer.transform(X_val)

# Scale the data (important for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Reduce dimensionality with PCA
pca = PCA(n_components=20)  # adjust the number of components based on your dataset
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)

# Model initialization (use LinearSVR for faster training)
svr_model = LinearSVR(C=1.0)  # Linear SVR for faster computation

# Train the model on the entire dataset
svr_model.fit(X_train_pca, y_train)

# Saving the model
joblib.dump(svr_model, 'models/svr_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(pca, 'models/pca.pkl')

# Validation prediction
y_pred = svr_model.predict(X_val_pca)

# Evaluation
mae = mean_absolute_error(y_val, y_pred)
rmse = mean_squared_error(y_val, y_pred, squared=False)
r2 = r2_score(y_val, y_pred)

# Print evaluation results
print(f'SVR Validation MAE: {mae}')
print(f'SVR Validation RMSE: {rmse}')
print(f'SVR Validation R²: {r2}')