# Import the experimental feature to use HalvingRandomSearchCV
from sklearn.experimental import enable_halving_search_cv  
from sklearn.model_selection import HalvingRandomSearchCV
import numpy as np
from sklearn.svm import SVR
import lightgbm as lgb
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import KFold

# Load data
X_train = np.load('prepared_data/X_train.npy')
y_train = np.load('prepared_data/y_train.npy')

# Reduce the dataset further (using an even smaller subset for faster tuning)
X_train_subset = X_train[:len(X_train) // 10]
y_train_subset = y_train[:len(y_train) // 10]

# Define custom scorer for negative mean squared error (as HalvingRandomSearchCV maximizes score)
scorer = make_scorer(mean_squared_error, greater_is_better=False)

# 1. SVM Hyperparameter Tuning with HalvingRandomSearchCV
svr = SVR()
param_grid_svr = {
    'kernel': ['rbf'],
    'C': [1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 0.01],
    'epsilon': [0.1, 0.2, 0.5]
}

# Use halving search for adaptive resource allocation
halving_search_svr = HalvingRandomSearchCV(
    svr, 
    param_distributions=param_grid_svr, 
    scoring=scorer, 
    cv=KFold(n_splits=2),  # Use 2-fold cross-validation for speed
    n_jobs=-1,  # Utilize all available CPU cores
    factor=2,  # Controls early stopping and resource allocation
    random_state=42
)
halving_search_svr.fit(X_train_subset, y_train_subset)
print(f'SVR Best Parameters: {halving_search_svr.best_params_}')
print(f'SVR Best Score: {-halving_search_svr.best_score_}')

# 2. LightGBM Hyperparameter Tuning with HalvingRandomSearchCV and early stopping
lgb_estimator = lgb.LGBMRegressor()
param_dist_lgbm = {
    'num_leaves': [31, 50, 70],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 500],
    'subsample': [0.6, 0.8, 1.0]
}

# Use halving search for LightGBM
halving_search_lgbm = HalvingRandomSearchCV(
    lgb_estimator, 
    param_distributions=param_dist_lgbm, 
    scoring=scorer, 
    cv=KFold(n_splits=2), 
    n_jobs=-1, 
    factor=2, 
    random_state=42
)
halving_search_lgbm.fit(X_train_subset, y_train_subset, eval_set=[(X_train_subset, y_train_subset)],
                        eval_metric='rmse', early_stopping_rounds=10, verbose=0)
print(f'LightGBM Best Parameters: {halving_search_lgbm.best_params_}')
print(f'LightGBM Best Score: {-halving_search_lgbm.best_score_}')
