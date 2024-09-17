# scripts/model_lightgbm.py
import matplotlib.pyplot as plt
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

lgb_estimator = lgb.LGBMRegressor(learning_rate=0.001, n_estimators=200, device='gpu')

# Rest of your code...

# Load data
X_train = np.load('prepared_data/X_train.npy')
y_train = np.load('prepared_data/y_train.npy')
X_val = np.load('prepared_data/X_val.npy')
y_val = np.load('prepared_data/y_val.npy')

# Create datasets
lgb_train = lgb.Dataset(X_train, y_train)
lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

# Parameters
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'verbose': -1
}

# Training
lgbm_model = lgb.train(params, lgb_train, valid_sets=[lgb_train, lgb_val], early_stopping_rounds=250)

# Saving the model
lgbm_model.save_model('models/lgbm_model.txt')

# Validation prediction
y_pred = lgbm_model.predict(X_val, num_iteration=lgbm_model.best_iteration)

# Evaluation
mae = mean_absolute_error(y_val, y_pred)
rmse = mean_squared_error(y_val, y_pred, squared=False)
r2 = r2_score(y_val, y_pred)
print(f'LightGBM Validation MAE: {mae}')
print(f'LightGBM Validation RMSE: {rmse}')
print(f'LightGBM Validation R²: {r2}')

lgb.plot_importance(lgbm_model, max_num_features=10)
plt.show()
