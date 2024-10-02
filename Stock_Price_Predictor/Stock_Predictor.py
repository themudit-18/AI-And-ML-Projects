import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

# Load the trained model
model = joblib.load('./model/stock_model.pkl')

st.title("Stock Price Prediction App")

# Load stock data
stock_data = pd.read_csv('./data/stock_data.csv')  # Update with your dataset path
stock_data['Date'] = pd.to_datetime(stock_data['Date'])

# Display available date range
min_date = stock_data['Date'].min()
max_date = stock_data['Date'].max()

st.write(f"Available data range: {min_date.date()} to {max_date.date()}")

# Get user input for prediction
symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, GOOG):", "AAPL")
start_date = st.date_input("Start Date", value=min_date.date())
end_date = st.date_input("End Date", value=max_date.date())

# Button to trigger prediction
if st.button("Predict"):
    # Filter stock data based on user input (date range)
    filtered_data = stock_data[(stock_data['Date'] >= pd.to_datetime(start_date)) & 
                               (stock_data['Date'] <= pd.to_datetime(end_date))]

    # Check if the filtered data is empty
    if filtered_data.empty:
        st.error(f"No data available for the selected date range: {start_date} to {end_date}. Please choose a different range.")
    else:
        # Extract features for the model
        X = filtered_data[['Open', 'High', 'Low', 'Volume']]
        
        # Apply scaling if necessary
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Generate predictions
        predictions = model.predict(X_scaled)
        
        # Add predictions to the dataframe
        filtered_data['Predicted Close'] = predictions
        
        # Calculate the difference (change) between actual and predicted closing prices
        filtered_data['Change'] = filtered_data['Predicted Close'] - filtered_data['Close']
        
        # Plot historical stock prices (candlestick chart)
        fig = go.Figure(data=[go.Candlestick(x=filtered_data['Date'],
                                             open=filtered_data['Open'],
                                             high=filtered_data['High'],
                                             low=filtered_data['Low'],
                                             close=filtered_data['Close'])])

        fig.update_layout(title=f'Historical Stock Prices for {symbol}',
                          xaxis_title='Date',
                          yaxis_title='Price (USD)')

        st.subheader("Historical Stock Prices")
        st.plotly_chart(fig, use_container_width=True)

        # Plot predicted vs actual stock prices
        fig2 = go.Figure(data=[go.Scatter(x=filtered_data['Date'], y=filtered_data['Predicted Close'], 
                                          mode='lines', name='Predicted Close')])

        fig2.add_trace(go.Scatter(x=filtered_data['Date'], y=filtered_data['Close'], 
                                  mode='lines', name='Actual Close', line=dict(dash='dash')))

        fig2.update_layout(title=f'Predicted vs Actual Stock Prices for {symbol}',
                           xaxis_title='Date',
                           yaxis_title='Price (USD)')

        st.subheader("Predicted vs Actual Stock Prices")
        st.plotly_chart(fig2, use_container_width=True)
        
        # Show change in stock price (predicted vs actual)
        st.subheader("Change in Stock Price (Predicted vs Actual)")
        st.write(filtered_data[['Date', 'Close', 'Predicted Close', 'Change']])
        
        # Highlight the changes with conditional formatting
        st.dataframe(filtered_data[['Date', 'Close', 'Predicted Close', 'Change']].style.applymap(
            lambda x: 'background-color: lightgreen' if x > 0 else 'background-color: lightcoral', 
            subset=['Change']))
