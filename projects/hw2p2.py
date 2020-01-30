import json
import itertools
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn import metrics
​
def get_metrics(y_true, y_pred):
    metrics_dict = {}
    metrics_dict["MAE"] = metrics.mean_absolute_error(y_true, y_pred)
    metrics_dict["MSE"] = metrics.mean_squared_error(y_true, y_pred)
    metrics_dict["R2"] = metrics.r2_score(y_true, y_pred)
    metrics_dict["MAPE"] = np.mean(np.abs(y_pred - y_true)/y_true)
    return metrics_dict
​
# Reading the data
data = pd.read_csv("AAPL.csv")
data["Date"] = pd.to_datetime(data["Date"])
data = data.drop(columns=["Adj Close", "Low", "High", "Open"])
data = data.sort_index()
​
# Add the target
data["y"] = data["Close"].shift(-1) - data["Close"]
​
# Add engineered features
data["vol"] = np.log(data["Volume"]+1)
data["vol_lag1"] = data["vol"].shift(1)
data["vol_lag2"] = data["vol"].shift(2)
data["vol_lag3"] = data["vol"].shift(3)
data["close_sma1"] = data["Close"].rolling(30).mean()/data["Close"]
data["vol_sma1"] = data["vol"].rolling(30).mean()/data["vol"]
data["close_sma2"] = data["Close"].rolling(120).mean()/data["Close"]
data["vol_sma2"] = data["vol"].rolling(120).mean()/data["vol"]
​
data = data.dropna()
data_train = data[data["Date"] <= "2019-04-30"]
data_train_train = data_train[data_train["Date"] <= "2019-03-30"]
data_train_test = data_train[data_train["Date"] > "2019-03-30"]
data_test = data[data["Date"] > "2019-04-30"]
y_train_train = data_train_train["y"]
y_train_test = data_train_test["y"]
y_test = data_test["y"]
X_train_train = data_train_train.drop(columns=["y", "Date"])
X_train_test = data_train_test.drop(columns=["y", "Date"])
X_test = data_test.drop(columns=["y", "Date"])
​
# Finding the best hyperparameters for training dataset
res_best = 1e10
model_best = None
alpha_lst = [1e-3, 1e-2, 1e-1]
l1_ratio_lst = [0.1, 0.5]
for alpha, l1_ratio in itertools.product(alpha_lst, l1_ratio_lst):
​
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio,
                        tol=1e-2, max_iter=1e5,
                        normalize=True, fit_intercept=False)
    model.fit(X_train_train, y_train_train)
    y_pred = model.predict(X_train_test) + X_train_test["Close"].values
    y_true = y_train_test.values + X_train_test["Close"].values
    res = get_metrics(y_true, y_pred)["MSE"]
    if  res < res_best:
        res_best = res
        model_best = model
​
y_true = y_test.values + X_test["Close"].values
y_pred = model_best.predict(X_test) + X_test["Close"].values
y_base = X_test["Close"].values
​
print("Eval[MyModel]:")
print(json.dumps(get_metrics(y_true, y_pred), indent=2))
print("Eval[Baseline]:")
print(json.dumps(get_metrics(y_true, y_base), indent=2))