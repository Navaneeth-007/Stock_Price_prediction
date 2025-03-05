import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import streamlit as st
import matplotlib.pyplot as plt

# Add a background image using the provided URL
st.markdown("""
    <style>
    .stApp {
        background-image: url('https://img.freepik.com/premium-vector/futuristic-stock-market-background-with-trend-graph_83282-38.jpg');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        height: 100vh;
    }
    </style>
    """, unsafe_allow_html=True)


# List of stock symbols to show in the dropdown (you can expand this list or fetch dynamically)
stock_symbols = ['AAPL', 'GOOGL', 'AMZN', 'MSFT', 'TSLA', 'META', 'NFLX', 'SPY']

# Streamlit UI
st.title('Stock Price Prediction')


# Dropdown for stock selection
stock_symbol = st.selectbox('Select Stock Symbol:', stock_symbols)

# Restrict date input fields between 2010 and June 2024
min_date = pd.to_datetime('2010-01-01')
max_date = pd.to_datetime('2024-06-30')

# Date input fields
start_date = st.date_input('Start Date', min_value=min_date, max_value=max_date, value=min_date)
end_date = st.date_input('End Date', min_value=start_date, max_value=max_date, value=max_date)

if st.button('Run Prediction'):
    # Fetch the stock data using yfinance
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    data = data[['Close']]

    # Data Preprocessing
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Prepare data for LSTM
    def create_dataset(dataset, time_step=100):
        X, y = [], []
        for i in range(len(dataset) - time_step - 1):
            X.append(dataset[i:(i + time_step), 0])
            y.append(dataset[i + time_step, 0])
        return np.array(X), np.array(y)

    # Split data into train/test datasets
    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    test_data = scaled_data[train_size:]

    time_step = 100
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Build and train LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=1, epochs=1)

    # Predictions
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Inverse transformation
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    # Plot the results
    train = data[:train_size + time_step + 1]
    valid = data[train_size + time_step + 1:]
    valid['Predictions'] = test_predict

    # Create a figure and axes for the plot
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_title('Stock Price Prediction Model')
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price USD ($)')
    ax.plot(train['Close'], label='Training Data')
    ax.plot(valid['Close'], label='Actual Price')
    ax.plot(valid['Predictions'], label='Predicted Price')
    ax.legend(loc='lower right')

    # Show the plot in Streamlit
    st.pyplot(fig)

    print(data.head())
