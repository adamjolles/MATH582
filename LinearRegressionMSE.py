import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime, timedelta

# Download stock data
stock_symbol = 'AAPL'  # Example with Apple stock
end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)
stock_data = yf.download(stock_symbol, start, end)

# Prepare data for linear regression
stock_data.reset_index(inplace=True)
stock_data['Days'] = (stock_data['Date'] - stock_data['Date'].min()).dt.days

# Define the predictor (independent variable) and response (dependent variable)
X = stock_data[['Days']]
y = stock_data['Close']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the testing set
y_pred = model.predict(X_test)

# Calculate and print MSE
mse = mean_squared_error(y_test, y_pred)
print(f"The Mean Squared Error (MSE) on the test set is: {mse}")

# Add predictions to the DataFrame for existing data
stock_data['Predicted'] = model.predict(X)

# Predict future prices
future_days = 30  # Predicting for 30 days into the future
last_day = stock_data['Days'].iloc[-1]
future_dates = [last_day + i for i in range(1, future_days + 1)]
future_prices = model.predict(np.array(future_dates).reshape(-1, 1))

# Print future predictions
for i, price in enumerate(future_prices, start=1):
    future_date = end + timedelta(days=i)
    print(f"Predicted price for {future_date.strftime('%Y-%m-%d')} is {price:.2f}")

# Plot the results with future predictions
plt.figure(figsize=(12, 6))
plt.scatter(stock_data['Date'], stock_data['Close'], color='blue', label='Actual Close Price')
plt.plot(stock_data['Date'], stock_data['Predicted'], color='red', label='Predicted Close Price')

# Add future predictions to the plot
future_pred_dates = [end + timedelta(days=i) for i in range(1, future_days + 1)]
plt.plot(future_pred_dates, future_prices, color='green', linestyle='dashed', label='Future Predictions')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title(f'Linear Regression Analysis and Future Predictions of {stock_symbol} Stock')
plt.legend()
plt.show()
