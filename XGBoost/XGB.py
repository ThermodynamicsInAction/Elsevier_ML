#import libraries
import numpy as np, pandas as pd, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
import os
from sklearn.metrics import make_scorer, mean_squared_error
import sklearn

df = pd.read_csv('dataset.csv',encoding = 'utf8',sep=';')
X = df[['M_C', 'M_A', 'SYM', 'P', 'T']].values
y = df[['EXP U']].values

'''Data preparation - dividing the data into a training and test set'''
X_train, X_test, y_train, y_test = train_test_split(
X,y,test_size = 0.4, random_state = 42)
'''Scaling features for better model performance'''
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
y_train = y_train.ravel()
y_test = y_test.ravel()

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'objective': 'reg:squarederror',  # Task: regression
    'max_depth': 4,                  # Max. of tree depth
    'learning_rate': 0.3,
    'random_state': 42
}
##To do - n
xgb_model = xgb.train(params, dtrain, num_boost_round=300)

'''Model training'''
y_test_xgb = xgb_model.predict(dtest)
y_train_xgb = xgb_model.predict(dtrain)

'''Metrics'''
r2 = r2_score(y_test, y_test_xgb)
print(f"Determination coefficient R^2: {r2}")
mse = mean_squared_error(y_test, y_test_xgb)
print(f"MSE: {mse}")
r2_train = r2_score(y_train, y_train_xgb)
print(f"Determination coefficient R^2: {r2_train}")
mse_train = mean_squared_error(y_train, y_train_xgb)
print(f"MSE: {mse_train}")

'''Cross Validation'''
cv_scores = cross_val_score(xgb.XGBRegressor(**params), X_train, y_train, cv=25, scoring='r2')
mean_r2 = cv_scores.mean()
print("Mean R^2 after cross validation:", mean_r2)

cv_scores = cross_val_score(xgb.XGBRegressor(**params), X_test, y_test, cv=25, scoring='r2')
cv_scores_mse = cross_val_score(xgb.XGBRegressor(**params), X_test, y_test, cv=25, scoring='neg_mean_squared_error')


mean_r2 = cv_scores.mean()
mean_mse = cv_scores_mse.mean()
print("Mean R^2 after cross validation:", mean_r2)
print("Mean MSE after cross validation:", -mean_mse)

test_predictions_xgb = pd.Series(y_test_xgb)
pred_df = pd.DataFrame(y_test,columns = ['Test TRUE Y'])
pred_df = pd.concat([pred_df,test_predictions_xgb],axis = 1)
pred_df.columns = ['Test true y', 'Pred']

train_predictions_xgb = pd.Series(y_train_xgb)
train_df = pd.DataFrame(y_train,columns = ['Test TRUE Y'])
train_df = pd.concat([train_df,train_predictions_xgb],axis = 1)
train_df.columns = ['Test true y', 'Pred']

sns.scatterplot(x = 'Test true y', y = 'Pred', data = train_df)
sns.scatterplot(x = 'Test true y', y = 'Pred', data = pred_df, alpha = 0.2)

import os
directory_path = 'res'
if not os.path.exists(directory_path):
    os.makedirs(directory_path)
train_df.to_csv(os.path.join(directory_path, 'train_set_XB_U.csv'), sep=';', encoding='utf-8')
pred_df.to_csv(os.path.join(directory_path, 'test_set_XB_U.csv'), sep=';', encoding='utf-8')

'''ILS calculation scheme'''
def predictions3(MC,MA,SYM,P,T):
    res = []
    for j in T:
        for i in P:
            new_geom = [[MC,MA,SYM,i,j]]
            new_geom = scaler.transform(new_geom)
            new_geom_dmatrix = xgb.DMatrix(new_geom)  # Konwersja na DMatrix
            res.append(xgb_model.predict(new_geom_dmatrix))
            #print(model.predict(new_geom))
    return res
'''Example'''
il_name = '2Hea_Pr'
Mcat = 62.061
Man = 73.071

P = [0.10,1.00,2.00,3.00,4.00,5.00,6.00,7.00,8.00,9.00,10.00,11.00,12.00,13.00,14.00,15.00,16.00,17.00,18.00,19.00,20.00]
T = [303.15,313.15,323.15,333.15,343.15,353.15

]
result = predictions3(Mcat,Man,0,P,T);

'''Save to file as table'''
res_flat = np.array(result).flatten()
res_numerical = [val.item() for val in res_flat]
directory_path = il_name
if not os.path.exists(directory_path):
    os.makedirs(directory_path)
data_table= np.array(res_numerical).reshape(len(T), len(P))
data = pd.DataFrame(data_table)
data = data.T
data.to_excel(directory_path+'/'+il_name+'_U_DATA_XB.xlsx', index=False)