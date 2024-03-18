#import libraries
import numpy as np, pandas as pd, seaborn as sns

#-------Data load and preprocessing--------#
df = pd.read_csv('zbior_23.csv',encoding = 'utf8',sep=';')
X = df[['M_C', 'M_A', 'SYM', 'P', 'T']].values
y = df[['EXP U']].values
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
'''Przygotowanie danych - podzielenie danych na zbiór trenujacy i testowy'''
'''TO DO FUNCTIONAL test_size = n, random_state=num'''
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3, random_state = 42)
'''Scaler function'''
scaler = MinMaxScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
'''Model'''
from tensorflow.keras.models import Sequential
import tensorflow as tf
from keras.layers import Dense, Dropout
from keras.initializers import glorot_uniform

def custom_loss1(y_true, y_pred):
    regularization = 0.001  # Wartość regularyzacji, możesz dostosować ją do potrzeb
    l2_loss = 0
    for w in model.trainable_weights:
        l2_loss += regularization * tf.reduce_sum(tf.square(w))
    total_loss = tf.reduce_mean(tf.square(y_true - y_pred)) + l2_loss

    return total_loss

model = Sequential()
#model.add(Dense(5,activation = 'tanh'))
#model = Sequential()
model.add(Dense(5, activation='tanh', kernel_initializer=glorot_uniform(), input_shape=(X_train.shape[1],)))
model.add(Dense(25, activation='gelu', kernel_initializer=glorot_uniform()))
model.add(Dense(25, activation='gelu', kernel_initializer=glorot_uniform()))
model.add(Dense(25, activation='gelu', kernel_initializer=glorot_uniform()))
model.add(Dense(25, activation='gelu', kernel_initializer=glorot_uniform()))
model.add(Dense(25, activation='gelu', kernel_initializer=glorot_uniform()))
model.add(Dense(25, activation='gelu', kernel_initializer=glorot_uniform()))
model.add(Dense(25, activation='gelu', kernel_initializer=glorot_uniform()))
model.add(Dense(25, activation='gelu', kernel_initializer=glorot_uniform()))
model.add(Dense(1, kernel_initializer=glorot_uniform()))

model.compile(optimizer='adam', loss=custom_loss1)
from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs=4000, validation_data=(X_test, y_test))

loss_df = pd.DataFrame(model.history.history)
loss_df.plot()

model.evaluate(X_test,y_test)
test_predictions = model.predict(X_test)
train_predictions = model.predict(X_train)

test_predictions = pd.Series(test_predictions.reshape(860,))
train_predictions1 = pd.Series(train_predictions.reshape(2005,))
pred_df = pd.DataFrame(y_test,columns = ['Test TRUE Y'])
pred_df = pd.concat([pred_df,test_predictions],axis = 1)
pred_df.columns = ['Test true y', 'Pred']

train_df = pd.DataFrame(y_train,columns = ['Test TRUE Y'])
train_df = pd.concat([train_df,train_predictions1],axis = 1)
train_df.columns = ['Test true y', 'Pred']

sns.scatterplot(x = 'Test true y', y = 'Pred', data = train_df)
sns.scatterplot(x = 'Test true y', y = 'Pred', data = pred_df, alpha = 0.2)

'''Save train and test dataframe to CSV'''
train_df.to_csv('train_set.csv', sep=';', encoding='utf-8')
pred_df.to_csv('test_set.csv', sep=';', encoding='utf-8')

def predictions3(MC,MA,SYM,P,T):
    res = []
    for j in T:
        for i in P:
            new_geom = [[MC,MA,SYM,i,j]]
            new_geom = scaler.transform(new_geom)
            res.append(model.predict(new_geom))
            #print(model.predict(new_geom))
    return res

il_name = 'C4C1Pyr_NTF2'
Mcat = 127.137
Man = 280.146

P = [0.1,5,10,20,30,40,50,60,70,80,90,95]
T = [293.15,303.15,313.15,323.15,333.15,343.15,353.15,363.15,373.15,393.15,413.15

]
result = predictions3(Mcat,Man,0,P,T);
res_flat = np.array(result).flatten()
res_numerical = [val.item() for val in res_flat]

'''Create Excel file with speed of sound predicted data'''
data_u_table = np.array(res_numerical).reshape(len(T), len(P))
data_u = pd.DataFrame(data_u_table)
data_u = data_u.T
data_u.to_excel(il_name+'_U_DATA.xlsx', index=False)

model.save("UNN_GELU_1.h5")

'''USE pretrained model'''
from tensorflow.keras.models import load_model
model_test = load_model("UNN_GELU_1.h5", custom_objects={'custom_loss1': custom_loss1})
train_predictionsPretModel = model_test.predict(X_train)
'''NN Summary'''
from tensorflow.keras.utils import plot_model
model_test.summary()
plot_model(model,to_file='UNN_GELU.png',show_shapes=True)



