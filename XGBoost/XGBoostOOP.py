import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from os import makedirs  # Assuming Python 3
import numpy as np
from xgboost import XGBoostRegressor


class DataPreprocessor:
    def __init__(self, file_path, encoding='utf-8', sep=';'):
        self.file_path = file_path
        self.encoding = encoding
        self.sep = sep

    def load_data(self):
        # Load data from CSV using pandas
        self.df = pd.read_csv(self.file_path, encoding=self.encoding, sep=self.sep)

    def split_data(self, test_size=0.4, random_state=42):
        # Split data into training and testing sets
        X = self.df[['M_C', 'M_A', 'SYM', 'P', 'T']].values
        y = self.df[['EXP U']].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)

    def scale_data(self):
        # Create scaler instance within the class
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
def main():
    # Define model parameters
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 4,
        'learning_rate': 0.3,
        'random_state': 42
    }

    # Load and preprocess data
    preprocessor = DataPreprocessor('zbior_23.csv')  # Assuming your file path
    preprocessor.load_data()
    preprocessor.split_data()
    preprocessor.scale_data()

    # Create and train XGBoost model
    model = XGBoostRegressor(params)
    model.train_model(preprocessor.X_train, preprocessor.y_train)

    # Create ILS calculator
    ils_calculator = ILScalculator(model)

    # Example usage
    il_name = '2Hea_Pr'
    Mcat = 62.061
    Man = 73.071
    P = [0.10, 1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00, 8.00, 9.00, 10.00, 11.00, 12.00, 13.00, 14.00, 15.00, 16.00, 17.00, 18.00, 19.00, 20.00]
    T = [303.15, 313.15, 323.15, 333.15, 343.15, 353.15]

    results = ils_calculator.calculate_ils(Mcat, Man, 0, P, T, preprocessor.scaler)  # Pass scaler
    ils_calculator.save_results_to_excel(results, il_name)

    # You can add more example usages here or call the ILS calculator for different data points

if __name__ == "__main__":
    main()