import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import seaborn as sns
import os
from sklearn.base import BaseEstimator

class XGBoostModel:
    def __init__(self, params):
        self.params = params
        self.model = None

    def train(self, X_train, y_train, num_boost_round=300):
        dtrain = xgb.DMatrix(X_train, label=y_train)
        self.model = xgb.train(self.params, dtrain, num_boost_round=num_boost_round)

    def predict(self, X):
        dmatrix = xgb.DMatrix(X)
        return self.model.predict(dmatrix)

class XGBoostSklearnAdapter(BaseEstimator):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return self.model.predict(X)

def load_data(file_path):
    return pd.read_csv(file_path, encoding='utf8', sep=';')

def save_results(data, file_name):
    directory_path = 'results'
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    data.to_csv(os.path.join(directory_path, file_name), sep=';', encoding='utf-8', index=False)

def prepare_data(df, target_column='EXP U', test_size=0.4, random_state=42):
    X = df[['M_C', 'M_A', 'SYM', 'P', 'T']].values
    y = df[[target_column]].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    y_train = y_train.ravel()
    y_test = y_test.ravel()

    return X_train, X_test, y_train, y_test

def evaluate_model(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return r2, mse

def cross_validate(model, X, y, cv=25, scoring='r2'):
    adapter_model = XGBoostSklearnAdapter(model)
    cv_scores = cross_val_score(adapter_model, X, y, cv=cv, scoring=scoring)
    return cv_scores.mean()

def predictions3(MC, MA, SYM, P, T, scaler, xgb_model):
    res = []
    for j in T:
        for i in P:
            new_geom = [[MC, MA, SYM, i, j]]
            new_geom = scaler.transform(new_geom)
            new_geom_dmatrix = xgb.DMatrix(new_geom)  # Konwersja na DMatrix
            res.append(xgb_model.predict(new_geom_dmatrix))
            #print(model.predict(new_geom))
    return res

def main(scaler):
    df = load_data('zbior_23.csv')
    X_train, X_test, y_train, y_test = prepare_data(df)

    params = {
        'objective': 'reg:squarederror',
        'max_depth': 4,
        'learning_rate': 0.3,
        'random_state': 42
    }

    model = XGBoostModel(params)
    model.train(X_train, y_train)

    # Dopasowanie scaler do danych treningowych
    scaler.fit(X_train)

    # Skalowanie danych treningowych i testowych
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    y_test_pred = model.predict(X_test_scaled)
    y_train_pred = model.predict(X_train_scaled)

    r2_test, mse_test = evaluate_model(y_test, y_test_pred)
    r2_train, mse_train = evaluate_model(y_train, y_train_pred)

    print(f"Test R^2: {r2_test}, Test MSE: {mse_test}")
    print(f"Train R^2: {r2_train}, Train MSE: {mse_train}")

    mean_r2_cv = cross_validate(model, X_train_scaled, y_train)
    print("Mean R^2 after cross-validation:", mean_r2_cv)

    # Saving results
    test_results = pd.DataFrame({'Test true y': y_test, 'Pred': y_test_pred})
    train_results = pd.DataFrame({'Train true y': y_train, 'Pred': y_train_pred})
    save_results(test_results, 'test_set_predictions.csv')
    save_results(train_results, 'train_set_predictions.csv')

    # Scatter plots
    test_predictions_xgb = pd.Series(y_test_pred)
    pred_df = pd.DataFrame(y_test, columns=['Test TRUE Y'])
    pred_df = pd.concat([pred_df, test_predictions_xgb], axis=1)
    pred_df.columns = ['Test true y', 'Pred']

    train_predictions_xgb = pd.Series(y_train_pred)
    train_df = pd.DataFrame(y_train, columns=['Test TRUE Y'])
    train_df = pd.concat([train_df, train_predictions_xgb], axis=1)
    train_df.columns = ['Test true y', 'Pred']

    sns.scatterplot(x='Test true y', y='Pred', data=train_df)
    sns.scatterplot(x='Test true y', y='Pred', data=pred_df, alpha=0.2)

    # Save scatter plots to CSV
    directory_path = 'res'
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    train_df.to_csv(os.path.join(directory_path, 'train_set_XB_U.csv'), sep=';', encoding='utf-8')
    pred_df.to_csv(os.path.join(directory_path, 'test_set_XB_U.csv'), sep=';', encoding='utf-8')

    # ILS calculation scheme
    il_name = '2Hea_Pr'
    Mcat = 62.061
    Man = 73.071

    P = [0.10,1.00,2.00,3.00,4.00,5.00,6.00,7.00,8.00,9.00,10.00,11.00,12.00,13.00,14.00,15.00,16.00,17.00,18.00,19.00,20.00]
    T = [303.15,313.15,323.15,333.15,343.15,353.15]

    result = predictions3(Mcat, Man, 0, P, T, scaler, model.model)

    # Save to file as table
    res_flat = np.array(result).flatten()
    res_numerical = [val.item() for val in res_flat]
    directory_path = il_name
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    tablica_danych = np.array(res_numerical).reshape(len(T), len(P))
    dane = pd.DataFrame(tablica_danych)
    dane = dane.T
    dane.to_excel(os.path.join(directory_path, il_name + '_U_DATA_XB.xlsx'), index=False)

if __name__ == "__main__":
    scaler = StandardScaler()  # Define scaler here or load it from somewhere else
    main(scaler)

