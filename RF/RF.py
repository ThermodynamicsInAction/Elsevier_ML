# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 08:48:17 2024

@author: user
"""
#import libraries
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from skopt import BayesSearchCV
import os

#%%
# Import dataset
df = pd.read_csv('zbior_23.csv', encoding='utf8', sep=';')
X = df[['M_C', 'M_A', 'SYM', 'P', 'T']].values
y = df[['EXP U']].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Scale features for better model performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#%% Default RF model
random_forest_model = RandomForestRegressor(n_estimators=10, random_state=42)
random_forest_model.fit(X_train, y_train.ravel())

# Make predictions and evaluate the default model
test_predictions_rf = random_forest_model.predict(X_test)
train_predictions_rf = random_forest_model.predict(X_train)

# Metrics for the test set
r2_test = r2_score(y_test, test_predictions_rf)
mse_test = mean_squared_error(y_test, test_predictions_rf)
print(f"R^2 coefficient (test): {r2_test}")
print(f"Mean Squared Error MSE (test): {mse_test}")

# Metrics for the training set
r2_train = r2_score(y_train, train_predictions_rf)
mse_train = mean_squared_error(y_train, train_predictions_rf)
print(f"R^2 coefficient (train): {r2_train}")
print(f"Mean Squared Error MSE (train): {mse_train}")

#%%
# Cross-validation with 25 folds
cv_scores = cross_val_score(random_forest_model, X_train, y_train.ravel(), cv=25, scoring='r2')
mean_r2 = cv_scores.mean()
print("Average R^2 after cross-validation:", mean_r2)

cv_scores_mse = cross_val_score(random_forest_model, X_train, y_train.ravel(), cv=25, scoring='neg_mean_squared_error')
mean_mse = cv_scores_mse.mean()
print("Average MSE after cross-validation:", -mean_mse)

cv_scores_test = cross_val_score(random_forest_model, X_test, y_test, cv=25, scoring='r2')
mean_r2_test = cv_scores_test.mean()
print("Average R^2 after cross-validation:", mean_r2_test)

#%%
# Generate plots
test_predictions_rf = pd.Series(test_predictions_rf)
pred_df = pd.DataFrame(y_test, columns=['Test TRUE Y'])
pred_df = pd.concat([pred_df, test_predictions_rf], axis=1)
pred_df.columns = ['Test true y', 'Pred']

train_predictions_rf = pd.Series(train_predictions_rf)
train_df = pd.DataFrame(y_train, columns=['Train TRUE Y'])
train_df = pd.concat([train_df, train_predictions_rf], axis=1)
train_df.columns = ['Train true y', 'Pred']

sns.scatterplot(x='Train true y', y='Pred', data=train_df)
sns.scatterplot(x='Test true y', y='Pred', data=pred_df, alpha=0.2)

#%%
# Define parameters for Grid Search
param_grid = {
    'n_estimators': [2,10,25,50],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

#%%
# Grid Search
grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42), param_grid=param_grid, scoring='neg_mean_squared_error', cv=5, verbose=1)
grid_search.fit(X_train, y_train.ravel())

print("Best parameters found by Grid Search:", grid_search.best_params_)

#%%
# Bayesian Optimization
from skopt.space import Real, Integer, Categorical
#%%
param_space = {
    'n_estimators': Integer(2, 300),
    'max_depth': Integer(1, 30),
    'min_samples_split': Integer(2, 10),
    'min_samples_leaf': Integer(1, 4),
    'bootstrap': Categorical([True, False])
}

#%%
bayes_search = BayesSearchCV(
    estimator=RandomForestRegressor(random_state=42), 
    search_spaces=param_space, 
    scoring='neg_mean_squared_error', 
    cv=5, 
    n_iter=50, 
    verbose=1, 
    random_state=42,
    n_points=3  # Number of starting points for random initialization
)
bayes_search.fit(X_train, y_train.ravel())

print("Best parameters found by Bayesian Optimization:", bayes_search.best_params_)

#%%
# Evaluate the model with the best parameters from Grid Search
best_model_grid = grid_search.best_estimator_

# Evaluate the model with the best parameters from Bayesian Optimization
best_model_bayes = bayes_search.best_estimator_

# Predict for test and train sets with the Grid Search model
y_pred_grid_test = best_model_grid.predict(X_test)
y_pred_grid_train = best_model_grid.predict(X_train)

# Predict for test and train sets with the Bayesian Optimization model
y_pred_bayes_test = best_model_bayes.predict(X_test)
y_pred_bayes_train = best_model_bayes.predict(X_train)

# Print predictions for comparison
print("Grid Search Test Predictions:", y_pred_grid_test[:10])  # Display first 10 predictions for brevity
print("Bayesian Optimization Test Predictions:", y_pred_bayes_test[:10])  # Display first 10 predictions for brevity

# Metrics for the test set
r2_grid_test = r2_score(y_test, y_pred_grid_test)
mse_grid_test = mean_squared_error(y_test, y_pred_grid_test)
rmse_grid_test = np.sqrt(mse_grid_test)

print(f"Grid Search - Test Set - R^2: {r2_grid_test}")
print(f"Grid Search - Test Set - MSE: {mse_grid_test}")
print(f"Grid Search - Test Set - RMSE: {rmse_grid_test}")

r2_bayes_test = r2_score(y_test, y_pred_bayes_test)
mse_bayes_test = mean_squared_error(y_test, y_pred_bayes_test)
rmse_bayes_test = np.sqrt(mse_bayes_test)

print(f"Bayesian Optimization - Test Set - R^2: {r2_bayes_test}")
print(f"Bayesian Optimization - Test Set - MSE: {mse_bayes_test}")
print(f"Bayesian Optimization - Test Set - RMSE: {rmse_bayes_test}")

# Metrics for the training set
r2_grid_train = r2_score(y_train, y_pred_grid_train)
mse_grid_train = mean_squared_error(y_train, y_pred_grid_train)
rmse_grid_train = np.sqrt(mse_grid_train)

print(f"Grid Search - Train Set - R^2: {r2_grid_train}")
print(f"Grid Search - Train Set - MSE: {mse_grid_train}")
print(f"Grid Search - Train Set - RMSE: {rmse_grid_train}")

r2_bayes_train = r2_score(y_train, y_pred_bayes_train)
mse_bayes_train = mean_squared_error(y_train, y_pred_bayes_train)
rmse_bayes_train = np.sqrt(mse_bayes_train)

print(f"Bayesian Optimization - Train Set - R^2: {r2_bayes_train}")
print(f"Bayesian Optimization - Train Set - MSE: {mse_bayes_train}")
print(f"Bayesian Optimization - Train Set - RMSE: {rmse_bayes_train}")

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

#%%
# Visualization for Grid Search
test_predictions_grid = pd.Series(y_pred_grid_test, name='Pred')
train_predictions_grid = pd.Series(y_pred_grid_train, name='Pred')

test_pred_df_grid = pd.DataFrame(y_test, columns=['Test TRUE Y']).join(test_predictions_grid)
train_pred_df_grid = pd.DataFrame(y_train, columns=['Train TRUE Y']).join(train_predictions_grid)

sns.scatterplot(x='Test TRUE Y', y='Pred', data=test_pred_df_grid, alpha=0.6).set_title("RF Grid")
sns.scatterplot(x='Train TRUE Y', y='Pred', data=train_pred_df_grid, alpha=0.2)

#%%
test_predictions_bayes = pd.Series(y_pred_bayes_test, name='Pred')
train_predictions_bayes = pd.Series(y_pred_bayes_train, name='Pred')

test_pred_df_bayes = pd.DataFrame(y_test, columns=['Test TRUE Y']).join(test_predictions_bayes)
train_pred_df_bayes = pd.DataFrame(y_train, columns=['Train TRUE Y']).join(train_predictions_bayes)

sns.scatterplot(x='Test TRUE Y', y='Pred', data=test_pred_df_bayes, alpha=0.6).set_title("RF Bayes")
sns.scatterplot(x='Train TRUE Y', y='Pred', data=train_pred_df_bayes, alpha=0.2)

#%%
# Define the relative directory path
directory_path = os.path.join('U')

# Create the directory if it doesn't exist
if not os.path.exists(directory_path):
    os.makedirs(directory_path)

# Save the dataframes to the specified directory
train_predictions_grid.to_csv(os.path.join(directory_path, 'train_set_RF_U_GRID.csv'), sep=';', encoding='utf-8', index=False)
test_predictions_grid.to_csv(os.path.join(directory_path, 'test_set_RF_U_GRID.csv'), sep=';', encoding='utf-8', index=False)
train_predictions_bayes.to_csv(os.path.join(directory_path, 'train_set_RF_U_Bayes.csv'), sep=';', encoding='utf-8', index=False)
test_predictions_bayes.to_csv(os.path.join(directory_path, 'test_set_RF_U_Bayes.csv'), sep=';', encoding='utf-8', index=False)

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
nazwa = '2Hea_Pr'
Mcat = 62.061
Man = 73.071

P = [0.10, 1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00, 10.00, 11.00, 12.00, 13.00, 14.00, 15.00, 16.00, 17.00, 18.00, 19.00, 20.00]
T = [303.15, 313.15, 323.15, 333.15, 343.15, 353.15]

res_grid, res_bayes = predictions3(Mcat, Man, 0, P, T, best_model_grid, best_model_bayes, scaler)

res_flat_grid = np.array(res_grid).flatten()
res_numerical_grid = [val.item() for val in res_flat_grid]  # Ensure numerical values   
res_flat_bayes = np.array(res_bayes).flatten()
res_numerical_bayes = [val.item() for val in res_flat_bayes]  # Ensure numerical values  

directory_path = os.path.join('U', nazwa)
if not os.path.exists(directory_path):
    os.makedirs(directory_path)
data_grid = np.array(res_numerical_grid).reshape(len(T), len(P))
data_bayes = np.array(res_numerical_bayes).reshape(len(T), len(P))
dataG = pd.DataFrame(data_grid)
dataB = pd.DataFrame(data_bayes)
dataG = dataG.T
dataB = dataB.T
file_pathG = os.path.join(directory_path, nazwa + '_U_DATA_RF_GRID.xlsx')
file_pathB = os.path.join(directory_path, nazwa + '_U_DATA_RF_BAYES.xlsx')
dataG.to_excel(file_pathG, index=False)
dataB.to_excel(file_pathB, index=False)
#%%
# import joblib
# # Save the model after Grid Search
# grid_model_path = os.path.join(directory_path, 'best_model_grid.pkl')
# joblib.dump(best_model_grid, grid_model_path)

# # Save the model after Bayesian Optimization
# bayes_model_path = os.path.join(directory_path, 'best_model_bayes.pkl')
# joblib.dump(best_model_bayes, bayes_model_path)

# print(f"Models saved to {directory_path}")

# # Save the scaler
# scaler_path = os.path.join(directory_path, 'scaler.pkl')
# joblib.dump(scaler, scaler_path)
#%%
import joblib
import numpy as np
import pandas as pd
import os

# Load the model after Grid Search
loaded_grid_model = joblib.load('best_model_grid.pkl')

# Load the model after Bayesian Optimization
loaded_bayes_model = joblib.load('best_model_bayes.pkl')

# Load the scaler
scaler = joblib.load('scaler.pkl')
#%%
# Prediction function
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
# Define variables
nazwa = 'C2ImC1OC8_NTF2'
Mcat = 239.376
Man = 280.146

P = [0.10,
9.81,
19.62,
29.43,
39.24,
49.05,
58.86,
68.67,
78.48,
88.29,
98.10,
107.91,
117.72,
127.53,
137.34,
147.15
]
T = [295.15,313.15,333.45,353.45,373.15
]

issym = 0
#%%
# Perform predictions with the loaded model
res_grid, res_bayes = predictions3(Mcat, Man, issym, P, T, loaded_grid_model, loaded_bayes_model, scaler)

# Reshape results into appropriate formats
res_flat_grid = np.array(res_grid).flatten()
res_numerical_grid = [val.item() for val in res_flat_grid]  # Ensure numerical values   
res_flat_bayes = np.array(res_bayes).flatten()
res_numerical_bayes = [val.item() for val in res_flat_bayes]  # Ensure numerical values  
#%%
# Cross-validated predictions for training set (using the best parameters from Grid Search)
y_pred_cv_grid_train = cross_val_predict(loaded_grid_model, X_train, y_train.ravel(), cv=5)

# Cross-validated predictions for training set (using the best parameters from Bayesian Optimization)
y_pred_cv_bayes_train = cross_val_predict(loaded_bayes_model, X_train, y_train.ravel(), cv=5)
#%%
directory_path = os.path.join('U',nazwa)
if not os.path.exists(directory_path):
    os.makedirs(directory_path)
data_grid = np.array(res_numerical_grid).reshape(len(T), len(P))
data_bayes = np.array(res_numerical_bayes).reshape(len(T), len(P))
dataG = pd.DataFrame(data_grid)
dataB = pd.DataFrame(data_bayes)
dataG = dataG.T
dataB = dataB.T
file_pathG = os.path.join(directory_path, nazwa + '_U_DATA_RF_GRID.xlsx')
file_pathB = os.path.join(directory_path, nazwa + '_U_DATA_RF_BAYES.xlsx')
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
file_pathG_smooth = os.path.join(directory_path, nazwa + '_U_DATA_RF_GRID_smooth.xlsx')
file_pathB_smooth = os.path.join(directory_path, nazwa + '_U_DATA_RF_BAYES_smooth.xlsx')
dataG_smooth.to_excel(file_pathG_smooth, index=False)
dataB_smooth.to_excel(file_pathB_smooth, index=False)

print(f"Smoothed data saved to {file_pathG_smooth} and {file_pathB_smooth}")

#%%
dataG_smooth = smooth_2d_data(dataG, degree=1)
dataB_smooth = smooth_2d_data(dataB, degree=1)
#%%
# Save the smoothed data to new Excel files
file_pathG_smooth = os.path.join(directory_path, nazwa + '_U_DATA_RF_GRID_smooth_1.xlsx')
file_pathB_smooth = os.path.join(directory_path, nazwa + '_U_DATA_RF_BAYES_smooth_1.xlsx')
dataG_smooth.to_excel(file_pathG_smooth, index=False)
dataB_smooth.to_excel(file_pathB_smooth, index=False)

print(f"Smoothed data saved to {file_pathG_smooth} and {file_pathB_smooth}")
#%%
print("End of script")
