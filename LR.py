# Install the dependencies
from sklearn.linear_model import LinearRegression
import math
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pylab import rcParams
from sklearn.svm import SVR


df = pd.read_csv("amd.us.txt")
df = df.drop("OpenInt",1)
#date to datetime
df.loc[:, 'Date'] = pd.to_datetime(df['Date'],format='%Y-%m-%d')
df.columns = [str(x).lower().replace(' ', '_') for x in df.columns]
#print(df.head(10))

#days = (df["date"]-df["date"][0]).dt.days
#df["days"] = days

#split data
num_val = int(0.3*len(df))
num_test = int(0.1*len(df))
num_train = len(df) - num_val - num_test

train = df[:num_train]
val = df[num_train:num_train+num_val]
test = df[num_train+num_val:]

#plot the data
rcParams['figure.figsize'] = 25, 10
ax = train.plot(x="date", y="close", style='b-', grid=True, fontsize=17)
ax = val.plot(x="date", y="close", style='gray', grid=True, ax=ax)
ax = test.plot(x="date", y="close", style='red', grid=True, ax=ax)

ax.legend(['train', 'validation', 'test'], fontsize=18)
ax.set_xlabel("Date", fontsize=25)
ax.set_ylabel("USD", fontsize=25)
plt.show()


dates = train.drop(["close"],1).drop(["date"],1).drop(["high"],1).drop(["low"],1).values
prices = train["close"].values

val_dates = val.drop(["close"],1).drop(["date"],1).drop(["high"],1).drop(["low"],1).values
val_prices = val["close"].values
test_dates = test.drop(["close"],1).drop(["date"],1).drop(["high"],1).drop(["low"],1).values

LinearReg = LinearRegression(fit_intercept=True).fit(dates, prices)
y_pred1 = LinearReg.predict(test_dates)
lin_score = LinearReg.score(val_dates,val_prices)
print("LinReg Score: ", lin_score)
test_price = test["close"].values

df2 = pd.DataFrame({"date":test["date"].values,"close":y_pred1})


