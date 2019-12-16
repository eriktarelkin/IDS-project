import math
import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pylab import rcParams
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

df = pd.read_csv("amd.us.txt")
df = df.drop("OpenInt",1)
#date to datetime
df.loc[:, 'Date'] = pd.to_datetime(df['Date'],format='%Y-%m-%d')
df.columns = [str(x).lower().replace(' ', '_') for x in df.columns]
#print(df.head(10))

days = (df["date"]-df["date"][0]).dt.days
df["days"] = days
#df = df.tail(5000)
print(len(df))
#split data
num_val = int(0.2*len(df))
num_test = int(0.2*len(df))
num_train = len(df) - num_val - num_test

train = df[:num_train]
val = df[num_train:num_train+num_val]
test = df[num_train+num_val:]

#plot the data
rcParams['figure.figsize'] = 10, 8
ax = train.plot(x="date", y="close", style='b-', grid=True)
ax = val.plot(x="date", y="close", style='y-', grid=True, ax=ax)
ax = test.plot(x="date", y="close", style='g-', grid=True, ax=ax)
ax.legend(['train', 'validation', 'test'])
ax.set_xlabel("date")
ax.set_ylabel("USD")
plt.show()


dates = train["days"].values
prices = train["close"].values
dates = np.reshape(dates, (len(dates), 1))
prices = np.reshape(prices, (len(prices),1))



val_dates = val["days"].values
val_prices = val["close"].values
val_dates = np.reshape(val_dates, (len(val_dates), 1))
val_prices = np.reshape(val_prices, (len(val_prices),1))

svr_rbf = LinearRegression()
svr_rbf.fit(dates,prices)

#svr_rbf = SVR(kernel="rbf", C=1e3, gamma=0.1)
#svr_rbf.fit(dates,prices)
#confidence score on validation data
svm_confidence = svr_rbf.score(val_dates,val_prices)
print("SVM confidence: ", svm_confidence)

test_dates = test["days"].values
test_dates = np.reshape(test_dates, (len(test_dates), 1))
predicted_stock = svr_rbf.predict(test_dates)

predicted_stock = [j for i in predicted_stock for j in i]

df2 = pd.DataFrame({"date":test["date"].values,"close":predicted_stock})

gx = test.plot(x="date", y="close", style='g-', grid=True)
gx = df2.plot(x="date", y="close", style="r-", grid = True, ax=gx)
gx.legend(['real', 'prediction'])
gx.set_xlabel("date")
gx.set_ylabel("USD")
plt.show()
