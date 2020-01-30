# Please do not use other libraries except for numpy
import numpy as np

class Ridge:

    def __init__(self):
        self.intercept = 0
        self.coef = None

    def fit(self, X, y, coef_prior=None, lmbd=1.0):
        n, m = X.shape
        self.coef = np.zeros(m)
        self.intercept = np.mean(y)
        I = np.eye(m)
        E = np.ones(n)
        if coef_prior == None:
            coef_prior = np.zeros(m)

        # a) normalize X
        # print(X)
        x_mu = np.mean(X, axis=0)
        x_sigma = np.sqrt(np.var(X, axis=0))
        X = (X - x_mu) / x_sigma  # normalized X

        # b) adjust coef_prior according to the normalization parameters
        coef_prior = coef_prior * x_sigma
        # print(coef_prior)

        # c) get coefficients
        ...
        XTX = np.linalg.inv(np.dot(X.T, X) + lmbd * I)
        # print(XTX.shape)
        # print(I.shape)
        self.intercept = self.intercept*E
        XTY = np.dot(X.T, y) + lmbd * coef_prior - np.dot(X.T, self.intercept)
        # print(self.intercept.shape)
        self.coef = np.dot(XTY, XTX.T)

        # d) adjust coefficients for de-normalized X
        # print(x_mu.shape)
        # print(self.coef.shape)
        self.intercept = self.intercept - np.dot(x_mu, self.coef / x_sigma)
        self.coef = self.coef / x_sigma

        return 0

    def get_coef(self):
        self.intercept = self.intercept[0]
        print("intercept is:")
        print(self.intercept)
        print("coef is:")
        print(self.coef)
        return self.intercept, self.coef


class ForwardStagewise:

    def __init__(self):
        self.intercept = 0
        self.path = []

    def fit(self, X, y, cannot_link=[], epsilon=1e-5, max_iter=1000):
        global beta1
        self.intercept = 0
        n, m = X.shape
        # a) normalize X
        y_mu = np.mean(y)
        x_mu = np.mean(X, axis=0)
        x_sigma = np.sqrt(np.var(X, axis=0))
        X = (X - x_mu) / x_sigma
        y = y - y_mu
        # b-1) implement incremental forwward-stagewise
        # b-2) implement cannot-link constraints
        beta = np.zeros(m)
        k = len(cannot_link)
        var_inactive = np.full(m, True)
        self.path.append(np.zeros([1, m]))
        for s in range(max_iter):
            # print(var_inactive)
            r = y - np.dot(X, beta)
            mse_min, j_best, gamma_best = np.inf, 0, 0
            # print(var_inactive)
            for j in np.where(var_inactive)[0]:
                gamma_j = np.dot(X[:, j], r) / np.dot(X[:, j], X[:, j])
                mse = np.mean(np.square(r - gamma_j * X[:, j]))
                if mse < mse_min:
                    mse_min, j_best, gamma_best = mse, j, gamma_j
            # print(j_best)
            for p in range(k):
                if j_best in cannot_link[p]:
                    q = len(cannot_link[p])
                    for i in range(q):
                        # print(cannot_link[p][i])
                        var_inactive[cannot_link[p][i]] = False
            if np.abs(gamma_best) > 0:
                beta[j_best] += gamma_best * epsilon
                # print(beta)
            # c) adjust coefficients for de-normalized X
            beta1 = beta / x_sigma
            self.intercept1 = self.intercept - np.dot(x_mu, beta / x_sigma) + y_mu
            # self.intercept1 = self.intercept - np.dot(x_mu, beta1)
            # d) construct the "path" numpy array
            #     path: l-by-m array,
            #               where l is the total number of iterations
            #               m is the number of features in X.
            #               The first row, path[0,:], should be all zeros.
            self.path.append(beta1)
        self.intercept = self.intercept1
        return 0

    def get_coef_path(self):
        print("intercept is:")
        print(self.intercept)
        print("path is:")
        print(self.path)
        return self.intercept, self.path
