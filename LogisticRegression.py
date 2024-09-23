import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime

# Download stock data
stock_symbol = 'AAPL'  # Example with Apple stock
end = datetime.now()
start = datetime(end.year - 1, end.month, end.day)
stock_data = yf.download(stock_symbol, start, end)

# Reset index to make 'Date' a column
stock_data.reset_index(inplace=True)

# Prepare data for logistic regression
stock_data['Price Change'] = stock_data['Close'].diff()
stock_data.dropna(inplace=True)  # Remove NaN values
stock_data['Target'] = np.where(stock_data['Price Change'] > 0, 1, 0)  # 1 for positive change, 0 for negative

# Define the predictor (independent variable) and response (dependent variable)
X = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]  # Using multiple features
y = stock_data['Target']

# Create and fit the model
model = LogisticRegression()
model.fit(X, y)

# Predict classes and probabilities
stock_data['Predicted Class'] = model.predict(X)
stock_data['Predicted Prob'] = model.predict_proba(X)[:, 1]

# Visualization
# Plot the predicted probabilities
plt.figure(figsize=(14, 7))
plt.plot(stock_data['Date'], stock_data['Predicted Prob'], color='green', label='Predicted Probability of Positive Change')

# Highlight the days where the actual class is '1' (positive change)
plt.scatter(stock_data['Date'][stock_data['Target'] == 1], 
            stock_data['Predicted Prob'][stock_data['Target'] == 1], 
            color='blue', 
            label='Actual Positive Change', 
            alpha=0.5)

plt.xlabel('Date')
plt.ylabel('Probability')
plt.title(f'Logistic Regression Predicted Probability of Positive Price Change for {stock_symbol}')
plt.legend()
plt.show()
