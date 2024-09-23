import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

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

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Transform the features into polynomial features
degree = 3  # Degree of the polynomial
poly_features = PolynomialFeatures(degree=degree)
X_poly_train = poly_features.fit_transform(X_train)
X_poly_test = poly_features.transform(X_test)

# Create and fit the model on the training set
model = LinearRegression()
model.fit(X_poly_train, y_train)

# Predict on the testing set
y_pred = model.predict(X_poly_test)

# Calculate and print MSE and R-squared
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"The Mean Squared Error (MSE) on the test set is: {mse}")
print(f"The R-squared score on the test set is: {r2}")

# Plot the results (using the full data for visualization)
X_poly = poly_features.fit_transform(X)
stock_data['Predicted'] = model.predict(X_poly)
plt.figure(figsize=(12, 6))
plt.scatter(stock_data['Date'], stock_data['Close'], color='blue', label='Actual Close Price')
plt.plot(stock_data['Date'], stock_data['Predicted'], color='red', label='Polynomial Regression Fit')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title(f'Polynomial Regression Analysis of {stock_symbol} Stock with degree {degree}')
plt.legend()
plt.show()
