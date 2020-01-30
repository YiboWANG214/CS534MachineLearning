import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import ElasticNet

from hw2 import LogisticBrier

# Problem 1
data = load_breast_cancer()
X = data.data
y = data.target
# loss_lst = [loss(y, X, beta)]

lb = LogisticBrier()
brier = lb.fit(X, y)
A = brier
print('the coefficients from the logistic regression for brier sore are:', A)

lr = LogisticRegression()
logistic = lr.fit(X, y)
B = logistic.coef_
print('the coefficients from the regular Logistic regression are:', B)

print(np.corrcoef(A, B))

# Problem 2

# 1.Download the Apple data form here
data = pd.read_csv('AAPL.csv')
n, m = np.shape(data)
# print(n,m)
# n = 1258, m = 7
y = data.loc[:, 'Close']
# print(data.describe())
# print(data.isnull().sum())
# data = data.dropna(axis=0)
# print(data)

# 3.Make various features that may help your predictive algorithms e.g. moving average
mean = []
var = []
median = []
change = []
percent_change = []
for i in range(1, n):
    mean.append(np.mean(y[0:i + 1]))
    var.append(np.var(y[0:i + 1]))
    median.append(np.median(y[0:i + 1]))
    change.append(y[i] - y[i - 1])
    percent_change.append((y[i] - y[i - 1]) / y[i - 1])
# print(np.shape(mean))
# print(np.shape(change))
mean = np.array(mean)
var = np.array(var)
median = np.array(median)
change = np.array(change)
percent_change = np.array(percent_change)

# 2.Your test set will be May, June, July of 2019. You will predict tomorrow’s closing price using past prices and
# volumes.
end = data[data.Date == '2019-05-01'].index.tolist()
# print(end)
# end = 1203

# trainDate = data.loc[0:1202, 'Date']
# print(trainDate)
trainMean = mean[0:1202]
trainVar = var[0:1202]
trainMedian = median[0:1202]
trainChange = change[0:1202]
trainPercentChange = percent_change[0:1202]

# trainX = None
trainX = np.column_stack((trainMean, trainVar, trainMedian, trainChange, trainPercentChange))
print(np.shape(trainX))
trainy = data.loc[2:1202, 'Close']
print(np.shape(trainy))

# testDate = data.loc[1203:, 'Date']
testMean = mean[1202:]
testVar = var[1202:]
testMedian = median[1202:]
testChange = change[1202:]
testPercentChange = percent_change[1202:]

# testX = None
testX = np.column_stack((testMean, testVar, testMedian, testChange, testPercentChange))
testy = data.loc[1204:, 'Close']


en = ElasticNet()
en.fit(trainX, trainy)
print(en.coef_)
print(en.score(testX, testy))
yy = en.predict(testX)
plt.plot(yy)
plt.plot(testy, 'ro')
# plt.show()
