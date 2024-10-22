import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import yfinance as yf

def get_stock_data(stock_symbol, start_date, end_date):
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    return data

def prepare_data(data, sequence_length):

    close_prices = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)
    
    x, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        x.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])
    
    x, y = np.array(x), np.array(y)
    x = np.reshape(x, (x.shape[0], x.shape[1], 1))
    return x, y, scaler

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dense(units=25))
    model.add(Dense(units=1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def plot_predictions(y_test_rescaled, predictions):
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_rescaled, label="Actual Price")
    plt.plot(predictions, label='Predicted Prices')
    plt.title("Stock Price Prediction")
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Parameters
    STOCK_SYMBOL = "ZOMATO.NS"
    START_DATE = "2015-01-01"
    END_DATE = "2024-01-01"
    SEQUENCE_LENGTH = 60
    BATCH_SIZE = 32
    EPOCHS = 10
    TEST_SIZE = 0.2

    # Get data
    data = get_stock_data(STOCK_SYMBOL, START_DATE, END_DATE)
    
    # Prepare data
    x, y, scaler = prepare_data(data, SEQUENCE_LENGTH)
    
    # Split data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, shuffle=False)
    
    # Build model
    model = build_lstm_model((x_train.shape[1], 1))
    
    # Train model
    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS)
    
    # Make predictions
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
    
    # Rescale actual data for comparison
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Plot results
    plot_predictions(y_test_rescaled, predictions)
