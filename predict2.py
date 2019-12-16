
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pylab import rcParams
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import math
from sklearn.linear_model import LinearRegression


df = pd.read_csv("amd.us.txt")
df = df.drop("OpenInt",1)
#date to datetime
df.loc[:, 'Date'] = pd.to_datetime(df['Date'],format='%Y-%m-%d')
df.columns = [str(x).lower().replace(' ', '_') for x in df.columns]
#print(df.head(10))


#days = (df["date"]-df["date"][0]).dt.days
#df["days"] = days
#df =df.tail(120)#only last year
#split data
num_val = int(0.2*len(df))
num_test = int(0.2*len(df))
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


dates = train.drop(["close"],1).drop(["date"],1).values
prices = train["close"].values

val_dates = val.drop(["close"],1).drop(["date"],1).values
val_prices = val["close"].values

print(dates,prices)
rfr = RandomForestRegressor().fit(dates,prices)
#confidence score on validation data
rfr_confidence = rfr.score(val_dates,val_prices)
print("rfr confidence: ", rfr_confidence)

test_dates = test.drop(["close"],1).drop(["date"],1).values

predicted_stock = rfr.predict(test_dates)
print(predicted_stock)

df2 = pd.DataFrame({"date":test["date"].values,"close":predicted_stock})

gx = train.plot(x="date", y="close", style='b-', grid=True)
gx = val.plot(x="date", y="close", style='y-', grid=True, ax=gx)
gx = test.plot(x="date", y="close", style='g-', grid=True, ax=gx)
gx = df2.plot(x="date", y="close", style="r-", grid = True, ax=gx)
gx.legend(["train","validation",'test', 'prediction'])
gx.set_xlabel("date")
gx.set_ylabel("USD")
plt.show()
