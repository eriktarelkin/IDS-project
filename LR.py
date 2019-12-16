# Install the dependencies
from sklearn.linear_model import LinearRegression
import math
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pylab import rcParams
from sklearn.svm import SVR


df = pd.read_csv("aapl.us.txt")
df = df.drop("OpenInt",1)
#date to datetime
df.loc[:, 'Date'] = pd.to_datetime(df['Date'],format='%Y-%m-%d')
df.columns = [str(x).lower().replace(' ', '_') for x in df.columns]
#print(df.head(10))

days = (df["date"]-df["date"][0]).dt.days
df["days"] = days

#split data
num_val = int(0.2*len(df))
num_test = int(0.2*len(df))
num_train = len(df) - num_val - num_test

train = df[:num_train]
val = df[num_train:num_train+num_val]
test = df[num_train+num_val:]
print(val)
print(train)
#plot the data
rcParams['figure.figsize'] = 25, 10
ax = train.plot(x="date", y="close", style='b-', grid=True, fontsize=17)
ax = val.plot(x="date", y="close", style='gray', grid=True, ax=ax)
ax = test.plot(x="date", y="close", style='red', grid=True, ax=ax)

ax.legend(['train', 'validation', 'test'], fontsize=18)
ax.set_xlabel("Date", fontsize=25)
ax.set_ylabel("USD", fontsize=25)
plt.show()


dates = train["days"].values
price = train["close"].values
dates = np.reshape(dates, (len(dates), 1))
price = np.reshape(price, (len(price), 1))

test_dates = test["days"].values
test_price = test["close"].values
test_dates = np.reshape(test_dates, (len(test_dates), 1))
test_price = np.reshape(test_price, (len(test_price), 1))


val_dates = val["days"].values
val_price = val["close"].values
val_dates = np.reshape(val_dates, (len(val_dates), 1))
val_price = np.reshape(val_price, (len(val_price), 1))

from sklearn.metrics import mean_squared_error
import math

LinearReg = LinearRegression(fit_intercept=True).fit(dates, price)
y_pred1 = LinearReg.predict(test_dates)
y_pred1 = [j for i in y_pred1 for j in i]
df2 = pd.DataFrame({"date":test["date"].values,"close":y_pred1})

gx = train.plot(x="date", y="close", style='b-', grid=True, fontsize = 17)
gx = val.plot(x="date", y="close", style='y-', grid=True, ax=gx)
gx = test.plot(x="date", y="close", style='g-', grid=True, ax=gx)
gx = df2.plot(x="date", y="close", style="r-", grid = True, ax=gx)
gx.legend(["train","validation",'test', 'prediction'], fontsize = 18)
gx.set_xlabel("Date", fontsize = 25)
gx.set_ylabel("USD", fontsize = 25)
plt.show()
RMSE = round(math.sqrt(mean_squared_error(test_price, y_pred1)), 2)
print("RMSE of the train model is: " + str(RMSE))
plt.show()
