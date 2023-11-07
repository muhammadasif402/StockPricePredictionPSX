import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
plt.style.use('fivethirtyeight')
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from PIL import Image
import yfinance as yf
import math
import joblib

# Function to implement RNN for stock price prediction
def rnn_predict(df, scaler, training_data_len):
    # ... (RNN implementation, similar to the LSTM part in your existing code)
    pass  # Modify this function as needed for RNN implementation

# Function to implement Linear Regression for stock price prediction
def linear_regression_predict(df, scaler, training_data_len):
    # ... (Linear Regression implementation)
    pass  # Modify this function as needed for Linear Regression implementation

# Streamlit app
def main():
    st.write('''
    # Stock Price Prediction Web Application
    Visualize and Predict The Stock Prices 
    ''')
    
    # Create a sidebar header
    st.sidebar.header('User Input')

    # Get user input for stock and date range
    stock_symbol = st.sidebar.text_input("Enter Stock Symbol ", "AAPL")  # Example stock symbol
    start_date = st.sidebar.text_input("Enter Start Date (YYYY-MM-DD)", "2012-01-01")
    end_date = st.sidebar.text_input("Enter End Date (YYYY-MM-DD)", "2019-12-17")

    # Download stock data using yfinance
    df = yf.download(stock_symbol, start=start_date, end=end_date)
    
    # Display stock data
    st.subheader('Stock Data')
    st.write(df)

    # Plot closing price history
    plt.figure(figsize=(16, 8))
    plt.plot(df['Close'])
    plt.title('Close Price History')
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD $', fontsize=18)
    st.subheader('Closing Price History')
    st.pyplot(plt)

    # Create a new DataFrame with only the 'Close' column
    data = df.filter(['Close'])

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Get the training data length
    training_data_len = math.ceil(len(scaled_data) * 0.8)

    # ... (existing LSTM code)

    # Add a dropdown for algorithm selection
    prediction_algorithm = st.sidebar.selectbox("Select Prediction Algorithm", ["LSTM", "RNN", "Linear Regression"])
    
    if prediction_algorithm == "LSTM":
        # ... (your existing LSTM code)
        pass
    elif prediction_algorithm == "RNN":
        rnn_predict(df, scaler, training_data_len)
    elif prediction_algorithm == "Linear Regression":
        linear_regression_predict(df, scaler, training_data_len)
    
    # ... (remaining code)

if __name__ == '__main__':
    st.set_option('deprecation.showPyplotGlobalUse', False)
    main()
