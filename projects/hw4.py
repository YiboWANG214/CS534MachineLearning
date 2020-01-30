import numpy as np
from sklearn.metrics import roc_auc_score


class GreedyKNN:

    def dist(self, x, X):
        distances = np.sqrt(np.sum(np.asarray(x - X) ** 2, axis=1))
        return distances

    def takeFirst(self, elem):
        return elem[0]

    def kNNpredict(self, X, X_train, y_train, k):

        n, m = np.shape(X)
        D = []
        y_hat = []

        for i in range(0, n):
            distances = self.dist(X[i, :], X_train)
            D = np.argsort(distances)
            sum_y = 0
            for t in range(1, k + 1):
                sum_y = sum_y + y_train[D[t]]
            y_mean = sum_y / k
            y_hat.append(y_mean)

        return y_hat

    def get_feature_order(self, X, y, k=5):

        n, m = np.shape(X)
        feature_lst = []

        while len(feature_lst) < m:
            max_auroc = 0.0
            max_var = -1
            for j in range(m):
                if j in feature_lst:
                    continue
                # Implement your own kNNpredict function
                # The function should return kNN prediction for the given X, y, and k
                y_hat = self.kNNpredict(X[:, feature_lst + [j]], X[:, feature_lst + [j]], y, k)
                auroc = roc_auc_score(y, y_hat)
                if auroc > max_auroc:
                    max_auroc = auroc
                    max_var = j
            feature_lst.append(max_var)

        return feature_lst
