import numpy as np
from scipy.special import expit
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
import warnings
​
​
class BrierLogisticRegression:
    def __init__(self, max_iter):
        self.intercept = 0
        self.coef_ = None
        self.max_iter = max_iter
        self.score = 1e10
​
    def loss(self, X, y, coef, intercept):
        return np.sum(np.square(y - intercept - expit(np.dot(X, coef))))
​
    def fit(self, X, y):
        self.coef_ = np.full(X.shape[1], 0)
        loss_old = self.loss(X, y, self.coef_, 0)
​
        print("Init. Loss = {}".format(loss_old))
        for i in range(self.max_iter):
            coef_old = self.coef_
            p = expit(np.dot(X, coef_old))
​
            # the first derivative of the loss function (Problem 1.1)
            gradient = -2 * np.dot(X.T,
                          (y - self.intercept - p) * p * (1 - p))
​
            # the Hessian of the loss function
            W = np.diag(-2 * (y - self.intercept -
                    2*(y - self.intercept + 1)*p + 3*p**2) * p * (1 - p))
            hessian = np.dot(np.dot(X.T, W), X)
​
            # Newton-Raphson Algorithm's update rule (Problem 1.2)
            # (Ref: http://diginole.lib.fsu.edu/islandora/object/fsu:360437/datastream/PDF/view, Page:40-41)
            tmp_coef = coef_old - np.dot(np.linalg.inv(hessian), gradient)
            tmp_intercept = (np.mean(y) -
                                np.mean(expit(np.dot(X, self.coef_))))
​
            loss_new = self.loss(X, y, tmp_coef, tmp_intercept)
            print("{} Iter. Loss = {}".format(i, loss_new))
            if loss_old - loss_new < 1e-5:
                break
            loss_old = loss_new
            self.coef_ = tmp_coef
            self.intercept = tmp_intercept
            self.score = loss_old
​
        print("Finished. Loss = {}".format(loss_old))
​
    def predict(self, X):
        y_hat = self.intercept + expit(np.dot(X, self.coef_))
        return y_hat
​
"""
Test
"""
data = load_breast_cancer()
X, y = data.data, data.target
cols = data.feature_names
​
blr = BrierLogisticRegression(max_iter=600)
blr.fit(X, y)
​
# Regularization parameters are set for parameter convergence
lr = LogisticRegression(penalty='l2', solver='newton-cg',
                        max_iter=10000, C=1e2)
lr.fit(X, y)
​
print("BrierScore of BrierLogistic: {}".format(blr.score))
score_lr = np.sum(np.square(y - lr.predict_proba(X)[:,1]))
print("BrierScore of RegularLogistic: {}".format(score_lr))
​
# Compare the coefficients from the regular Logistic regression and interpret the differences (Problem 1.4)
print("BrierLogistic coefficients:") # (Problem 1.3)
for k, v in zip(cols, blr.coef_):
    print("\t- {0}: {1:.2f}".format(k, v))
print("RegularLogistic coefficients:") # (Problem 1.3)
for k, v in zip(cols, lr.coef_[0]):
    print("\t- {0}: {1:.2f}".format(k, v))
​
"""
The main reason for the differences between the coefficients from the regular Logistic regression and ours is that the 
different loss functions are used. We use the Brier Score, while the regular Logistic regression uses the Bernoulli-form 
log-likelihood loss.
"""
​
# Bonus: compare the performance on a test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                            test_size=0.3, random_state=1)
blr.fit(X_train, y_train)
lr.fit(X_train, y_train)
y_hat_blr = blr.predict(X_test)
y_hat_lr = lr.predict_proba(X_test)[:,1]
score_blr = np.sum(np.square(y_test - y_hat_blr))
score_lr = np.sum(np.square(y_test - y_hat_lr))
print("BrierScore of BrierLogistic (test): {}".format(score_blr))
print("BrierScore of RegularLogistic (test): {}".format(score_lr))
​
​
# BrierLogistic is definitely overfitting.
# This is somewhat obvious from the "huge" coefficient values of the model
# Let's reduce the number of features, and compare:
X = X[:,:5]
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                            test_size=0.3, random_state=1)
blr.fit(X_train, y_train)
lr.fit(X_train, y_train)
y_hat_blr = blr.predict(X_test)
y_hat_lr = lr.predict_proba(X_test)[:,1]
score_blr = np.sum(np.square(y_test - y_hat_blr))
score_lr = np.sum(np.square(y_test - y_hat_lr))
print("BrierScore of BrierLogistic (test, reduced): {}".format(score_blr))
print("BrierScore of RegularLogistic (test, reduced): {}".format(score_lr))