import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
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

# Ensure there are no missing values
stock_data.dropna(inplace=True)

# Prepare data for logistic regression
stock_data['Price Change'] = stock_data['Close'].diff().fillna(0)
stock_data['Target'] = np.where(stock_data['Price Change'] > 0, 1, 0)

# Define the predictor (independent variable) and response (dependent variable)
X = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]  # Add your technical indicators here
y = stock_data['Target']

# Address Class Imbalance with SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Create a pipeline with scaling and logistic regression
pipeline = make_pipeline(StandardScaler(), LogisticRegression(C=1, class_weight='balanced'))

# Train the model
pipeline.fit(X_train, y_train)

# Predict on the test set
y_pred = pipeline.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
