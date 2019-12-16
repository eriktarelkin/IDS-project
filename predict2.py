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
print(df.head(10))


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
rcParams['figure.figsize'] = 10, 8
ax = train.plot(x="date", y="close", style='b-', grid=True)
ax = val.plot(x="date", y="close", style='y-', grid=True, ax=ax)
ax = test.plot(x="date", y="close", style='g-', grid=True, ax=ax)
ax.legend(['train', 'validation', 'test'])
ax.set_xlabel("date")
ax.set_ylabel("USD")
plt.show()

svr_rbf = SVR(kernel="rbf", C=1e3, gamma=0.1)
svr_rbf.fit(train,val)

svm_confidence = svr_rbf.score(train,test)
print("SVM confidence: ", svm_confidence)