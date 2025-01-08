import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
import math
import os


class StockPricePredictor:
    def __init__(self, stock_symbol: str, start: str, end: str, time_interval: int = 60):
        self.stock_symbol = stock_symbol
        self.start = start
        self.end = end
        self.time_interval = time_interval

        self.df = None
        self.scaler = None
        self.model = None
        self.X_train = self.y_train = None
        self.X_test = self.y_test = None
        self.y_pred_train = self.y_pred_test = None
            
    def load_data(self):
        """Fetch historical stock data."""
        self.df = yf.download(self.stock_symbol, self.start, self.end)
        self.df.reset_index(inplace=True)
        self.df.dropna(inplace=True)
        print("Data loaded successfully.")
    
    
    def scale_data(self, train_data: np.ndarray, test_data: np.ndarray):
        """Scale training and testing data."""
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_scaled = scaler.fit_transform(train_data)
        test_scaled = scaler.transform(test_data)
        return train_scaled, test_scaled, scaler
    
    def prepare_data(self):
        """Prepare training and testing datasets."""
        data = self.df["Close"].values.reshape(-1, 1)
        train_n = math.ceil(len(data) * 0.8)
        train_data, test_data = data[:train_n], data[train_n - self.time_interval:]
        
        train_scaled, test_scaled, self.scaler = self.scale_data(train_data, test_data)
        
        X_train, y_train = [], []
        X_test, y_test = [], test_scaled[self.time_interval:]
        
        for i in range(self.time_interval, len(train_scaled)):
            X_train.append(train_scaled[i - self.time_interval:i, 0])
            y_train.append(train_scaled[i, 0])
        
        for i in range(self.time_interval, len(test_scaled)):
            X_test.append(test_scaled[i - self.time_interval:i, 0])
        
        self.X_train = np.array(X_train).reshape(-1, self.time_interval, 1)
        self.y_train = np.array(y_train)
        self.X_test = np.array(X_test).reshape(-1, self.time_interval, 1)
        self.y_test = y_test
        print("Data prepared for training and testing.")
    
    def build_model(self):
        """Build the LSTM model."""
        self.model = Sequential([
            Input(shape=(self.X_train.shape[1], 1)),
            LSTM(50, return_sequences=True),
            LSTM(50, return_sequences=False),
            Dense(25),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        print("Model built successfully.")
    
    def train_model(self, epochs=10, batch_size=32):
        """Train the LSTM model."""
        history = self.model.fit(self.X_train, self.y_train, epochs=epochs, batch_size=batch_size, verbose=1)
        print("Model trained successfully.")
    
    def make_predictions(self):
        """Make predictions using the trained model."""
        self.y_pred_train = self.model.predict(self.X_train)
        self.y_pred_test = self.model.predict(self.X_test)
        
        self.y_pred_train = self.scaler.inverse_transform(self.y_pred_train.reshape(-1, 1))
        self.y_pred_test = self.scaler.inverse_transform(self.y_pred_test.reshape(-1, 1))
        self.y_test = self.scaler.inverse_transform(np.array(self.y_test).reshape(-1, 1))
        print("Predictions made successfully.")
    
    def evaluate_model(self):
        """Evaluate the model using Mean Squared Error."""
        mse_train = mean_squared_error(self.y_train, self.y_pred_train)
        mse_test = mean_squared_error(self.y_test, self.y_pred_test)
        print(f"Train MSE: {mse_train}")
        print(f"Test MSE: {mse_test}")
    
    def plot_results(self):
        """Plot the results."""
        train = self.df[:len(self.y_pred_train) + self.time_interval]
        valid = self.df[len(self.y_pred_train) + self.time_interval:]
        
        valid = valid.copy()
        valid["Predictions"] = self.y_pred_test.flatten()
        
        plt.figure(figsize=(16, 8))
        plt.title("Model Predictions")
        plt.xlabel("Date")
        plt.ylabel("Close Price USD ($)")
        plt.plot(train["Date"], train["Close"], label="Train")
        plt.plot(valid["Date"], valid["Close"], label="Validation")
        plt.plot(valid["Date"], valid["Predictions"], label="Predictions")
        plt.legend()
        plt.savefig("images/model.png")
        plt.show()
        print("Results plotted successfully.")
    
    def run_pipeline(self):
        """Run the full pipeline."""
        self.load_data()
        self.prepare_data()
        self.build_model()
        self.train_model()
        self.make_predictions()
        self.evaluate_model()
        self.plot_results()


# Run the pipeline
if __name__ == "__main__":
    predictor = StockPricePredictor(stock_symbol="AAPL", start="2012-01-01", end="2022-12-21")
    predictor.run_pipeline()
