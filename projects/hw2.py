import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt


class LogisticBrier:

    def __init__(self):
        self.intercept = 0
        self.beta = None

    def loss(self, X, y):
        ll = np.sum(np.square(y - self.intercept - expit(np.dot(X, self.beta))))
        return ll

    def fit(self, X, y, eps=1e-5):
        n, m = np.shape(X)
        if self.beta is None:
            self.beta = np.zeros(m)
            self.intercept = np.mean(y)
        loss_lst = [self.loss(X, y)]
        a = self.loss(X, y)
        b = 0
        while b < a:
            a = self.loss(X, y)
            loss_lst.append(self.loss(X, y))
            p = np.clip(expit(np.dot(X, self.beta)), eps, 1 - eps)
            W = np.diag(p * (1 - p))
            K = np.diag(p * p * (1 - p) * (y - self.intercept + 1))
            T = np.diag(p * p * p * (1 - p))
            Q = np.diag(p * (1 - p) * (y - self.intercept))
            # print(np.shape(K), np.shape(T), np.shape(W))
            # print(Q)
            # print(np.shape(Q))

            # 1.Derive the first derivative of the loss function
            derivative_first = -2 * np.dot(X.T, np.dot(y - self.intercept - p, W))

            # 2.Implement Newton’s method to solve the equation
            XTQX = np.dot(np.dot(X.T, Q), X)
            XTKX = np.dot(np.dot(X.T, K), X)
            XTTX = np.dot(np.dot(X.T, T), X)

            # print(np.shape(XTQX))
            # print(np.shape(XTWWX))
            derivative_second = -2 * (XTQX - 2 * XTKX + 3 * XTTX)
            # print(derivative_second)
            temp1 = self.beta - np.dot(np.linalg.inv(derivative_second), derivative_first)
            temp0 = np.mean(y - expit(np.dot(X, self.beta)))
            self.beta = temp1
            self.intercept = temp0
            b = self.loss(X, y)
        # print('the coefficients from the logistic regression for brier sore are:', self.beta)
        # print(self.intercept)
        print(loss_lst)
        plt.plot(loss_lst)
        # plt.show()
        return self.beta
