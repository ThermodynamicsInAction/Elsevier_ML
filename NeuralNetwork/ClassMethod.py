import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model

class DataHandler:
    def __init__(self, filename):
        self.df = pd.read_csv(filename, encoding='utf8', sep=';')

    def preprocess_data(self):
        X = self.df[['M_C', 'M_A', 'SYM', 'P', 'T']].values
        y = self.df[['EXP U']].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        scaler = MinMaxScaler()
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, y_train, y_test

class NeuralNetwork:
    def __init__(self):
        self.model = Sequential()
        self.build_model()

    def build_model(self):
        self.model.add(Dense(5, activation='tanh', kernel_initializer=glorot_uniform(), input_shape=(5,)))
        self.model.add(Dense(25, activation='gelu', kernel_initializer=glorot_uniform()))
        # Dodaj kolejne warstwy...
        self.model.compile(optimizer='adam', loss=self.custom_loss1)

    def custom_loss1(self, y_true, y_pred):
        regularization = 0.001
        l2_loss = 0
        for w in self.model.trainable_weights:
            l2_loss += regularization * np.sum(np.square(w))
        total_loss = np.mean(np.square(y_true - y_pred)) + l2_loss
        return total_loss

    def train_model(self, X_train, y_train, X_test, y_test):
        early_stopping = EarlyStopping(monitor='val_loss', mode='min', patience=10, restore_best_weights=True)
        history = self.model.fit(X_train, y_train, epochs=4000, validation_data=(X_test, y_test), callbacks=[early_stopping])

    def save_model(self, filename):
        self.model.save(filename)

    @staticmethod
    def load_model(filename, custom_objects):
        return load_model(filename, custom_objects=custom_objects)

class PredictionHandler:
    def __init__(self, scaler, model):
        self.scaler = scaler
        self.model = model

    def predict(self, MC, MA, SYM, P, T):
        res = []
        for j in T:
            for i in P:
                new_geom = [[MC, MA, SYM, i, j]]
                new_geom = self.scaler.transform(new_geom)
                res.append(self.model.predict(new_geom))
        return np.array(res).flatten().tolist()


if __name__ == "__main__":
    data_handler = DataHandler('zbior_23.csv')
    X_train, X_test, y_train, y_test = data_handler.preprocess_data()

    nn_model = NeuralNetwork()
    nn_model.train_model(X_train, y_train, X_test, y_test)
    nn_model.save_model("UNN_GELU_1.h5")

    prediction_handler = PredictionHandler(scaler, nn_model.model)
    il_name = 'C4C1Pyr_NTF2'
    Mcat = 127.137
    Man = 280.146
    P = [0.1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
    T = [293.15, 303.15, 313.15, 323.15, 333.15, 343.15, 353.15, 363.15, 373.15, 393.15, 413.15]
    result = prediction_handler.predict(Mcat, Man, 0, P, T)

    # Tworzenie pliku Excel z danymi
    data_u_table = np.array(result).reshape(len(T), len(P))
    data_u = pd.DataFrame(data_u_table)
    data_u = data_u.T
    data_u.to_excel(il_name+'_U_DATA.xlsx', index=False)

    # Użycie wytrenowanego modelu
    loaded_model = NeuralNetwork.load_model("UNN_GELU_1.h5", {'custom_loss1': nn_model.custom_loss1})
    train_predictionsPretModel = loaded_model.predict(X_train)

    # Wyświetlenie podsumowania modelu
    nn_model.model.summary()
    plot_model(nn_model.model, to_file='UNN_GELU.png', show_shapes=True)