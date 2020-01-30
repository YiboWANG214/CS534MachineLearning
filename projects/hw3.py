import numpy as np
import pandas as pd
import math
from numpy import nan
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_absolute_error


class QuarternaryDecisionTree:

    def __init__(self):
        self.tree = []
        self.split_var = []
        self.split_value = []
        self.split_mean = []
        self.predictions = []

    def select_split_pair(self, X, y):
        s1, s2, m1, m2 = 0, 0, 0, 0
        n, m = np.shape(X)
        InfoGain0 = -float('inf')
        for i in range(0, m):
            for t in range(i + 1, m):
                x1 = 0
                x2 = 0
                x3 = 0
                x4 = 0
                y1 = []
                y2 = []
                y3 = []
                y4 = []
                for j in np.arange(0.1, 1, 0.2):
                    for k in np.arange(0.1, 1, 0.2):
                        for p in range(0, n):
                            if X[p, i] > k and X[p, t] > j:
                                x1 = x1 + 1
                                y1 = np.append(y1, y[p])
                            if X[p, i] > k and X[p, t] <= j:
                                x2 = x2 + 1
                                y2 = np.append(y2, y[p])
                            if X[p, i] <= k and X[p, t] > j:
                                x3 = x3 + 1
                                y3 = np.append(y3, y[p])
                            if X[p, i] <= k and X[p, t] <= j:
                                x4 = x4 + 1
                                y4 = np.append(y4, y[p])
                        y1 = np.mean(y1)
                        y2 = np.mean(y2)
                        y3 = np.mean(y3)
                        y4 = np.mean(y4)
                        # print(y1, y2, y3, y4)
                        if y1 == 0 or y1 == 1 or np.isnan(y1):
                            H1 = 0
                        else:
                            H1 = - y1 * math.log(y1) - (1 - y1) * math.log(1 - y1)
                        if y2 == 0 or y2 == 1 or np.isnan(y2):
                            H2 = 0
                        else:
                            H2 = - y2 * math.log(y2) - (1 - y2) * math.log(1 - y2)
                        if y3 == 0 or y3 == 1 or np.isnan(y3):
                            H3 = 0
                        else:
                            H3 = - y3 * math.log(y3) - (1 - y3) * math.log(1 - y3)
                        if y4 == 0 or y4 == 1 or np.isnan(y4):
                            H4 = 0
                        else:
                            H4 = - y4 * math.log(y4) - (1 - y4) * math.log(1 - y4)
                        InfoGain = - x1 * H1 - x2 * H2 - x3 * H3 - x4 * H4
                        if InfoGain > InfoGain0:
                            s1 = k
                            s2 = j
                            m1 = i
                            m2 = t
                            InfoGain0 = InfoGain
        return s1, s2, m1, m2

    def fit(self, X, y, max_depth):
        n = len(X)
        if n < 3 or max_depth == 0:
            self.split_mean.append(np.mean(y))
            return np.mean(y)
        X1, X2, X3, X4 = [], [], [], []
        y1, y2, y3, y4 = [], [], [], []

        s1, s2, m1, m2 = self.select_split_pair(X, y)
        for i in range(0, n):
            if X[i, m1] > s1 and X[i, m2] > s2:
                X1.append(X[i, :])
                y1 = np.append(y1, y[i])
            if X[i, m1] > s1 and X[i, m2] <= s2:
                X2.append(X[i, :])
                y2 = np.append(y2, y[i])
            if X[i, m1] <= s1 and X[i, m2] > s2:
                X3.append(X[i, :])
                y3 = np.append(y3, y[i])
            if X[i, m1] <= s1 and X[i, m2] <= s2:
                X4.append(X[i, :])
                y4 = np.append(y4, y[i])
        self.split_var.append([m1, m2])
        self.split_value.append([s1, s2])
        X1 = np.array(X1)
        X2 = np.array(X2)
        X3 = np.array(X3)
        X4 = np.array(X4)
        self.tree = ({'split_var1': m1,
                      'split_var2': m2,
                      'split_value1': s1,
                      'split_value2': s2,
                      'X1': self.fit(X1, y1, max_depth - 1),
                      'X2': self.fit(X2, y2, max_depth - 1),
                      'X3': self.fit(X3, y3, max_depth - 1),
                      'X4': self.fit(X4, y4, max_depth - 1)})
        return self.tree, self.split_var, self.split_value, self.split_mean

    def predict(self, X):
        n = len(X)
        k = len(self.split_var)
        for i in range(n):
            m1 = self.split_var[0][0]
            m2 = self.split_var[0][1]
            if k >= 2:
                if X[i, m1] > self.split_value[0][0] and X[i, m2] > self.split_value[0][1]:
                    m3 = self.split_var[1][0]
                    m4 = self.split_var[1][1]
                    if X[i, m3] > self.split_value[1][0] and X[i, m4] > self.split_value[1][1]:
                        self.predictions.append(self.split_mean[0])
                    if X[i, m3] > self.split_value[1][0] and X[i, m4] <= self.split_value[1][1]:
                        self.predictions.append(self.split_mean[1])
                    if X[i, m3] <= self.split_value[1][0] and X[i, m4] > self.split_value[1][1]:
                        self.predictions.append(self.split_mean[2])
                    if X[i, m3] <= self.split_value[1][0] and X[i, m4] <= self.split_value[1][1]:
                        self.predictions.append(self.split_mean[3])
            if k >= 3:
                if X[i, m1] > self.split_value[0][0] and X[i, m2] <= self.split_value[0][1]:
                    m3 = self.split_var[2][0]
                    m4 = self.split_var[2][1]
                    if X[i, m3] > self.split_value[2][0] and X[i, m4] > self.split_value[2][1]:
                        self.predictions.append(self.split_mean[4])
                    if X[i, m3] > self.split_value[2][0] and X[i, m4] <= self.split_value[2][1]:
                        self.predictions.append(self.split_mean[5])
                    if X[i, m3] <= self.split_value[2][0] and X[i, m4] > self.split_value[2][1]:
                        self.predictions.append(self.split_mean[6])
                    if X[i, m3] <= self.split_value[2][0] and X[i, m4] <= self.split_value[2][1]:
                        self.predictions.append(self.split_mean[7])
            if k >= 4:
                if X[i, m1] <= self.split_value[0][0] and X[i, m2] > self.split_value[0][1]:
                    m3 = self.split_var[3][0]
                    m4 = self.split_var[3][1]
                    if X[i, m3] > self.split_value[3][0] and X[i, m4] > self.split_value[3][1]:
                        self.predictions.append(self.split_mean[8])
                    if X[i, m3] > self.split_value[3][0] and X[i, m4] <= self.split_value[3][1]:
                        self.predictions.append(self.split_mean[9])
                    if X[i, m3] <= self.split_value[3][0] and X[i, m4] > self.split_value[3][1]:
                        self.predictions.append(self.split_mean[10])
                    if X[i, m3] <= self.split_value[3][0] and X[i, m4] <= self.split_value[3][1]:
                        self.predictions.append(self.split_mean[11])
            if k >= 5:
                if X[i, m1] <= self.split_value[0][0] and X[i, m2] <= self.split_value[0][1]:
                    m3 = self.split_var[4][0]
                    m4 = self.split_var[4][1]
                    if X[i, m3] > self.split_value[4][0] and X[i, m4] > self.split_value[4][1]:
                        self.predictions.append(self.split_mean[12])
                    if X[i, m3] > self.split_value[4][0] and X[i, m4] <= self.split_value[4][1]:
                        self.predictions.append(self.split_mean[13])
                    if X[i, m3] <= self.split_value[4][0] and X[i, m4] > self.split_value[4][1]:
                        self.predictions.append(self.split_mean[14])
                    if X[i, m3] <= self.split_value[4][0] and X[i, m4] <= self.split_value[4][1]:
                        self.predictions.append(self.split_mean[15])
        return self.predictions

    def score(self, X, y):
        y_predict = self.predictions
        MSE = mean_squared_error(y, y_predict)
        MSLE = mean_squared_log_error(y, y_predict)
        MAE = mean_absolute_error(y, y_predict)
        return {"MSE:": MSE,
                "MSLE": MSLE,
                "MAE": MAE}


class DaRDecisionTree:

    def __init__(self):
        self.tree = {}

    def select_split_pair(self, X, y):
        s1, m1 = 0, 0
        n, m = np.shape(X)
        mse0 = float('inf')
        mse = float('inf')
        for i in range(0, m):
            X1 = []
            X2 = []
            y1 = []
            y2 = []
            step = (max(X[:, i]) - min(X[:, i])) / 10
            for j in np.arange(min(X[:, i]) + step, max(X[:, i]), step):
                for p in range(0, n):
                    if X[p, i] > j:
                        X1.append(X[p, :])
                        y1 = np.append(y1, y[p])
                    if X[p, i] <= j:
                        X2.append(X[p, :])
                        y2 = np.append(y2, y[p])
                if len(y1) is not 0 and len(y2) is 0:
                    y1_mean = [np.mean(y1)] * len(y1)
                    mse = mean_squared_error(y1, y1_mean)
                if len(y1) is 0 and len(y2) is not 0:
                    y2_mean = [np.mean(y2)] * len(y2)
                    mse = mean_squared_error(y2, y2_mean)
                if len(y1) is not 0 and len(y2) is not 0:
                    y1_mean = [np.mean(y1)] * len(y1)
                    y2_mean = [np.mean(y2)] * len(y2)
                    y_ = np.append(y1, y2)
                    y_mean = np.append(y1_mean, y2_mean)
                    mse = mean_squared_error(y_, y_mean)
                if mse < mse0:
                    s1 = j
                    m1 = i
                    mse0 = mse
        return s1, m1

    def fit(self, X, y, max_depth):
        n, m = X.shape
        if n < 3 or max_depth == 0:
            reg = Ridge()
            reg.fit(X, y)
            return reg
        s_value, j_best = self.select_split_pair(X, y)

        right_idx = X[:, j_best] > s_value
        left_idx = X[:, j_best] <= s_value

        X_left, y_left = X[left_idx, :], y[left_idx]
        X_right, y_right = X[right_idx, :], y[right_idx]

        self.tree = ({'split_var': j_best,
                      'split_value': s_value,
                      'left': self.fit(X_left, y_left, max_depth - 1),
                      'right': self.fit(X_right, y_right, max_depth - 1)})
        return self.tree

    def find_leaf_node(self, X, Y):
        if X[Y['split_var']] <= Y['split_value']:
            if isinstance(Y['left'], dict):
                return self.find_leaf_node(X, Y['left'])
            else:
                return Y['left'].predict(np.array([X]))
        else:
            if isinstance(Y['right'], dict):
                return self.find_leaf_node(X, Y['right'])
            else:
                return Y['right'].predict(np.array([X]))

    def predict(self, X):
        n = len(X)
        predictions = []
        for i in range(n):
            pred = self.find_leaf_node(X[i], self.tree)
            predictions.append(pred)
        return predictions

    def score(self, X, y):
        y_predict = self.predict(X)
        MSE = mean_squared_error(y, y_predict)
        MSLE = mean_squared_log_error(y, y_predict)
        MAE = mean_absolute_error(y, y_predict)
        return {"MSE:": MSE,
                "MSLE": MSLE,
                "MAE": MAE}
