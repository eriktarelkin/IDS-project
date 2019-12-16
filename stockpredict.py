# Install the dependencies
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Get the stock data
df = pd.read_csv("amd.us.txt", index_col=0)
# No need for openint
df = df.drop(["OpenInt"], 1)

#print(df.head())

# Variable for predicting "n" days into the future
days = 7

df = df[["Close"]]
# Another column shifted n units up
df["Prediction"] = df[["Close"]].shift(-days)
#print(df.tail())

# New data set
# Dataframe to a numpy array
print(df)
X = np.array(df.drop(["Prediction"], 1))

# Remove the last "n" rows
X = X[:-days]
print(X)

# New dependent data set
y = np.array(df["Prediction"])

y = y[:-days]
print(y)

# Splitting the data into 80% train 20% test

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create and train the Support Vector Regression
print("values")
print(x_train)
print(y_train)
"""
svr_rbf = SVR(kernel="rbf", C=1e5, gamma=0.1)
svr_rbf.fit(x_train, y_train)

# Testing Model: Score returns the coefficient of determination R^2 of the prediction.
# The best possible score is 1.0

svm_confidence = svr_rbf.score(x_test, y_test)
print("SVM confidence: ", svm_confidence)
x_forecast = np.array(df.drop(["Prediction"],1))[-days:]
#prediction n days in the future
predicted_stock = svr_rbf.predict(x_forecast)

df.loc[df.tail(days).index, 'Prediction'] = predicted_stock
print(df.tail(days))
ax = df.tail(days).plot(grid=True)
ax.set_xlabel("date")
ax.set_ylabel("USD")
plt.show()
"""


