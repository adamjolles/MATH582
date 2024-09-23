import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Download stock data
stock_symbol = 'AAPL'  # Example with Apple stock
end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)
stock_data = yf.download(stock_symbol, start, end)

# Prepare data for polynomial regression
stock_data.reset_index(inplace=True)
stock_data['Days'] = (stock_data['Date'] - stock_data['Date'].min()).dt.days

# Define the predictor and response variables
X = stock_data[['Days']]
y = stock_data['Close']

# Transform the features into polynomial features
degree = 3  # Degree of the polynomial
poly_features = PolynomialFeatures(degree=degree)
X_poly = poly_features.fit_transform(X)

# Create and fit the model
model = LinearRegression()
model.fit(X_poly, y)

# Predict using polynomial regression
stock_data['Predicted'] = model.predict(X_poly)

# Plot the results
plt.figure(figsize=(12, 6))
plt.scatter(stock_data['Date'], stock_data['Close'], color='blue', label='Actual Close Price')
plt.plot(stock_data['Date'], stock_data['Predicted'], color='red', label='Polynomial Regression Fit')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title(f'Polynomial Regression Analysis of {stock_symbol} Stock with degree {degree}')
plt.legend()
plt.show()
