# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 09:51:19 2024

@author: user
"""
#import libraries
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from skopt import BayesSearchCV
import os

#%%
df = pd.read_csv('zbior_23.csv', encoding='utf8', sep=';')
X = df[['M_C', 'M_A', 'SYM', 'P', 'T']].values
y = df[['EXP U']].values

#%%
'''Data preparation - splitting into training and test sets'''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

print(X_train.shape)
print(y_train.shape)
print('Test shapes')
print(X_test.shape)
print(y_test.shape)

#%%
# Feature scaling for better model performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#%%
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

#%%
params = {
    'objective': 'reg:squarederror',  # Task: regression
    'max_depth': 4,                   # Maximum tree depth
    'learning_rate': 0.3,
    'random_state': 42        
}

#%%
xgb_model = xgb.train(params, dtrain, num_boost_round=300)

#%%
# Predict and evaluate the default model
y_test_xgb = xgb_model.predict(dtest)
y_train_xgb = xgb_model.predict(dtrain)

#%%
# Metrics for the test set
r2 = r2_score(y_test, y_test_xgb)
print(f"R^2 coefficient of determination (test): {r2}")
mse = mean_squared_error(y_test, y_test_xgb)
print(f"Mean squared error MSE (test): {mse}")

#%%
# Metrics for the training set
r2_train = r2_score(y_train, y_train_xgb)
print(f"R^2 coefficient of determination (train): {r2_train}")
mse_train = mean_squared_error(y_train, y_train_xgb)
print(f"Mean squared error MSE (train): {mse_train}")

#%%
# Cross-validation with 25 folds
cv_scores = cross_val_score(xgb.XGBRegressor(**params), X_train, y_train, cv=25, scoring='r2')
mean_r2 = cv_scores.mean()
print("Average R^2 after cross-validation:", mean_r2)

#%%
cv_scores_mse = cross_val_score(xgb.XGBRegressor(**params), X_test, y_test, cv=25, scoring='neg_mean_squared_error')
mean_r2 = cv_scores.mean()
mean_mse = cv_scores_mse.mean()
print("Average R^2 after cross-validation:", mean_r2)
print("Average MSE after cross-validation:", -mean_mse)

#%%
# Grid Search for hyperparameter tuning
xgb_model_tuning = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

param_grid = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [3, 4, 5, 9],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0]
}

grid_search = GridSearchCV(estimator=xgb_model_tuning, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, verbose=1)
grid_search.fit(X_train, y_train)

print("Best parameters found by Grid Search:", grid_search.best_params_)

#%%
# Evaluate the model with the best parameters from Grid Search
best_model_grid = grid_search.best_estimator_

#%%
# Predict for test and train sets
y_pred_grid_test = best_model_grid.predict(X_test)
y_pred_grid_train = best_model_grid.predict(X_train)

# Metrics for the test set
r2_grid_test = r2_score(y_test, y_pred_grid_test)
mse_grid_test = mean_squared_error(y_test, y_pred_grid_test)
rmse_grid_test = np.sqrt(mse_grid_test)

print(f"Grid Search - Test Set - R^2: {r2_grid_test}")
print(f"Grid Search - Test Set - MSE: {mse_grid_test}")
print(f"Grid Search - Test Set - RMSE: {rmse_grid_test}")

# Metrics for the training set
r2_grid_train = r2_score(y_train, y_pred_grid_train)
mse_grid_train = mean_squared_error(y_train, y_pred_grid_train)
rmse_grid_train = np.sqrt(mse_grid_train)

print(f"Grid Search - Train Set - R^2: {r2_grid_train}")
print(f"Grid Search - Train Set - MSE: {mse_grid_train}")
print(f"Grid Search - Train Set - RMSE: {rmse_grid_train}")

#%%
# Visualization
test_predictions_grid = pd.Series(y_pred_grid_test, name='Pred')
train_predictions_grid = pd.Series(y_pred_grid_train, name='Pred')

test_pred_df = pd.DataFrame(y_test, columns=['Test TRUE Y']).join(test_predictions_grid)
train_pred_df = pd.DataFrame(y_train, columns=['Train TRUE Y']).join(train_predictions_grid)

sns.scatterplot(x='Test TRUE Y', y='Pred', data=test_pred_df, alpha=0.6)
sns.scatterplot(x='Train TRUE Y', y='Pred', data=train_pred_df, alpha=0.2)

#%%Bayesian Optimization
bayes_search = BayesSearchCV(estimator=xgb_model_tuning, search_spaces=param_grid, scoring='neg_mean_squared_error', cv=5, n_iter=32, verbose=1, random_state=42)
bayes_search.fit(X_train, y_train)

print("Best parameters found by Bayesian Optimization:", bayes_search.best_params_)

#%%
# Evaluate the model with the best parameters from Bayesian Optimization
best_model_bayes = bayes_search.best_estimator_

#%%
# Predict for test and train sets
y_pred_bayes_test = best_model_bayes.predict(X_test)
y_pred_bayes_train = best_model_bayes.predict(X_train)

# Metrics for the test set
r2_bayes_test = r2_score(y_test, y_pred_bayes_test)
mse_bayes_test = mean_squared_error(y_test, y_pred_bayes_test)
rmse_bayes_test = np.sqrt(mse_bayes_test)

print(f"Bayesian Optimization - Test Set - R^2: {r2_bayes_test}")
print(f"Bayesian Optimization - Test Set - MSE: {mse_bayes_test}")
print(f"Bayesian Optimization - Test Set - RMSE: {rmse_bayes_test}")

# Metrics for the training set
r2_bayes_train = r2_score(y_train, y_pred_bayes_train)
mse_bayes_train = mean_squared_error(y_train, y_pred_bayes_train)
rmse_bayes_train = np.sqrt(mse_bayes_train)

print(f"Bayesian Optimization - Train Set - R^2: {r2_bayes_train}")
print(f"Bayesian Optimization - Train Set - MSE: {mse_bayes_train}")
print(f"Bayesian Optimization - Train Set - RMSE: {rmse_bayes_train}")

#%%
directory_path = os.path.join('U')
if not os.path.exists(directory_path):
    os.makedirs(directory_path)
metrics_path = os.path.join(directory_path, 'metrics.txt')
with open(metrics_path, 'w') as f:
    f.write("Grid Search - Test Set - R^2: {:.4f}\n".format(r2_grid_test))
    f.write("Grid Search - Test Set - MSE: {:.4f}\n".format(mse_grid_test))
    f.write("Grid Search - Test Set - RMSE: {:.4f}\n".format(rmse_grid_test))
    f.write("Bayesian Optimization - Test Set - R^2: {:.4f}\n".format(r2_bayes_test))
    f.write("Bayesian Optimization - Test Set - MSE: {:.4f}\n".format(mse_bayes_test))
    f.write("Bayesian Optimization - Test Set - RMSE: {:.4f}\n".format(rmse_bayes_test))
    f.write("Grid Search - Train Set - R^2: {:.4f}\n".format(r2_grid_train))
    f.write("Grid Search - Train Set - MSE: {:.4f}\n".format(mse_grid_train))
    f.write("Grid Search - Train Set - RMSE: {:.4f}\n".format(rmse_grid_train))
    f.write("Bayesian Optimization - Train Set - R^2: {:.4f}\n".format(r2_bayes_train))
    f.write("Bayesian Optimization - Train Set - MSE: {:.4f}\n".format(mse_bayes_train))
    f.write("Bayesian Optimization - Train Set - RMSE: {:.4f}\n".format(rmse_bayes_train))

print(f"Metrics saved to {metrics_path}")

#%%
# Prediction visualization
test_predictions_bayes = pd.Series(y_pred_bayes_test, name='Pred')
train_predictions_bayes = pd.Series(y_pred_bayes_train, name='Pred')

test_pred_df_bayes = pd.DataFrame(y_test, columns=['Test TRUE Y']).join(test_predictions_bayes)
train_pred_df_bayes = pd.DataFrame(y_train, columns=['Train TRUE Y']).join(train_predictions_bayes)

sns.scatterplot(x='Test TRUE Y', y='Pred', data=test_pred_df_bayes, alpha=0.6)
sns.scatterplot(x='Train TRUE Y', y='Pred', data=train_pred_df_bayes, alpha=0.2)

#%%
# Save results
directory_path = os.path.join('results_new', 'xboost', 'U')

if not os.path.exists(directory_path):
    os.makedirs(directory_path)

train_predictions_grid.to_csv(os.path.join(directory_path, 'train_set_XB_U_GRID.csv'), sep=';', encoding='utf-8', index=False)
test_predictions_grid.to_csv(os.path.join(directory_path, 'test_set_XB_U_GRID.csv'), sep=';', encoding='utf-8', index=False)
train_predictions_bayes.to_csv(os.path.join(directory_path, 'train_set_XB_U_Bayes.csv'), sep=';', encoding='utf-8', index=False)
test_predictions_bayes.to_csv(os.path.join(directory_path, 'test_set_XB_U_Bayes.csv'), sep=';', encoding='utf-8', index=False)

#%%
def predictions3(MC, MA, SYM, P, T, model_grid, model_bayes, scaler):
    res_grid = []
    res_bayes = []

    for j in T:
        for i in P:
            new_geom = [[MC, MA, SYM, i, j]]
            new_geom = scaler.transform(new_geom)
            
            # Predict with Grid Search model
            pred_grid = model_grid.predict(new_geom)
            res_grid.append(pred_grid)
            
            # Predict with Bayesian Optimization model
            pred_bayes = model_bayes.predict(new_geom)
            res_bayes.append(pred_bayes)
    
    return res_grid, res_bayes

#%%
nazwa = 'C4Mim_BF4'
Mcat = 139.123
Man = 88.805

P = [0.1, 5, 10, 20, 30, 45, 60, 80, 100, 125, 151]
T = [283.15, 293.15, 303.15, 313.15, 323.15]

issym = 0

#%%
res_grid, res_bayes = predictions3(Mcat, Man, issym, P, T, best_model_grid, best_model_bayes, scaler)

res_flat_grid = np.array(res_grid).flatten()
res_numerical_grid = [val.item() for val in res_flat_grid]  
res_flat_bayes = np.array(res_bayes).flatten()
res_numerical_bayes = [val.item() for val in res_flat_bayes]

#%%
directory_path = os.path.join('results_new', 'xboost', 'U', nazwa)
if not os.path.exists(directory_path):
    os.makedirs(directory_path)
data_grid = np.array(res_numerical_grid).reshape(len(T), len(P))
data_bayes = np.array(res_numerical_bayes).reshape(len(T), len(P))
dataG = pd.DataFrame(data_grid)
dataB = pd.DataFrame(data_bayes)
dataG = dataG.T
dataB = dataB.T
file_pathG = os.path.join(directory_path, nazwa + '_U_DATA_XB_GRID.xlsx')
file_pathB = os.path.join(directory_path, nazwa + '_U_DATA_XB_BAYES.xlsx')
dataG.to_excel(file_pathG, index=False)
dataB.to_excel(file_pathB, index=False)

#%%
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Function to smooth 2D data using polynomial regression
def smooth_2d_data(data, degree=3):
    # Prepare meshgrid for pressure (rows) and temperature (columns)
    Y, X = np.meshgrid(np.arange(data.shape[0]), np.arange(data.shape[1]), indexing='ij')
    
    # Flatten the grids and the data
    X_flat = X.ravel()
    Y_flat = Y.ravel()
    Z_flat = data.values.ravel()
    
    # Prepare polynomial features
    poly = PolynomialFeatures(degree)
    XY_poly = poly.fit_transform(np.column_stack([X_flat, Y_flat]))
    
    # Fit the polynomial model
    poly_reg = LinearRegression().fit(XY_poly, Z_flat)
    
    # Predict smoothed values
    Z_smooth_flat = poly_reg.predict(XY_poly)
    
    # Reshape the smoothed values back to the original data shape
    Z_smooth = Z_smooth_flat.reshape(data.shape)
    
    return pd.DataFrame(Z_smooth, index=data.index, columns=data.columns)

# Smooth dataG and dataB using a polynomial of degree 3
dataG_smooth = smooth_2d_data(dataG, degree=2)
dataB_smooth = smooth_2d_data(dataB, degree=2)

#%%
# Save the smoothed data to new Excel files
file_pathG_smooth = os.path.join(directory_path, nazwa + '_U_DATA_XB_GRID_smooth.xlsx')
file_pathB_smooth = os.path.join(directory_path, nazwa + '_U_DATA_XB_BAYES_smooth.xlsx')
dataG_smooth.to_excel(file_pathG_smooth, index=False)
dataB_smooth.to_excel(file_pathB_smooth, index=False)

print(f"Smoothed data saved to {file_pathG_smooth} and {file_pathB_smooth}")

#%%
dataG_smooth = smooth_2d_data(dataG, degree=1)
dataB_smooth = smooth_2d_data(dataB, degree=1)

#%%
# Save the smoothed data to new Excel files
file_pathG_smooth = os.path.join(directory_path, nazwa + '_U_DATA_XB_GRID_smooth_1.xlsx')
file_pathB_smooth = os.path.join(directory_path, nazwa + '_U_DATA_XB_BAYES_smooth_1.xlsx')
dataG_smooth.to_excel(file_pathG_smooth, index=False)
dataB_smooth.to_excel(file_pathB_smooth, index=False)

print(f"Smoothed data saved to {file_pathG_smooth} and {file_pathB_smooth}")
