import numpy as np

class Ridge:
    def __init__(self):
        self.intercept = 0
        self.coef = None

    def fit(self, X, y, coef_prior=None, lmbd=1.0):
        n, m = X.shape

        # For compatibility with sklearn
        lmbd = lmbd * n

        if coef_prior is None:
            coef_prior = np.zeros((m,))

        # a) normalize X
        x_mu = np.mean(X, axis=0)
        x_sigma = np.std(X, axis=0)
        X = (X - x_mu) / x_sigma
        y_intercept = np.mean(y)

        # b) adjust coef_prior according to the normalization parameters
        coef_prior = coef_prior * x_sigma

        # c) get coefficients
        I = np.identity(m)
        term_1 = np.linalg.inv(np.dot(X.T, X) + lmbd * I)
        term_2 = np.dot(X.T, y) + lmbd * coef_prior
        self.coef = np.dot(term_1, term_2)

        # d) adjust coefficients for de-normalized X
        self.intercept = y_intercept - np.sum(self.coef*x_mu / x_sigma)
        self.coef = self.coef / x_sigma

    def get_coef(self):
        return self.intercept, self.coef


class ForwardStagewise:

    def __init__(self):
        self.intercept = 0
        self.path = []

    def fit(self, X, y, cannot_link=[], epsilon=1e-1, max_iter=1000):

        # a) normalize X
        x_mu = np.mean(X, axis=0)
        x_sigma = np.std(X, axis=0)
        X = (X - x_mu) / x_sigma

        # b-1) implement incremental forward-stagewise
        #       (Refer: https://arxiv.org/pdf/0705.0269.pdf, page.5)
        # b-2) implement cannot-link constraints
        # c) adjust coefficients for de-normalized X
        # d) construct the "path" numpy array
        #     path: l-by-m array,
        #               where l is the total number of iterations
        #               m is the number of features in X.
        #               The first row, path[0,:], should be all zeros.
        n, m = X.shape
        self.path.append(np.zeros(m))
        self.intercept = np.mean(y)
        y = y - np.mean(y)
        beta = np.zeros(m)  # initialize the coefficients
        r = y - np.dot(X, beta)  # initialize the residuals

        # create a map that connects each variable to its associated
        #       cannot link variables
        from collections import defaultdict
        from itertools import combinations
        cannot_map = defaultdict(set)
        for group in cannot_link:
            for j, k in combinations(group, 2):
                cannot_map[j].add(k)
                cannot_map[k].add(j)
        deactivated_set = set()

        for s in range(max_iter):
            mse_min, j_best, gamma_best = np.inf, -1, 0

            # find the predictor x_j most correlated with r
            #   ignoring the deactivated variables
            for j in range(m):
                if j in deactivated_set:
                    continue
                gamma_j = np.dot(X[:, j], r) / np.dot(X[:, j], X[:, j])
                mse = np.mean(np.square(r - gamma_j * X[:, j]))
                if mse < mse_min:
                    mse_min, j_best, gamma_best = mse, j, gamma_j

            if j_best > -1:
                for k in cannot_map[j_best]:
                    deactivated_set.add(k)

                delta = epsilon * np.sign(gamma_best)
                beta[j_best] += delta # update beta_j
                r -= delta * X[:, j_best] # update residual
            else:
                break

            self.path.append(beta)

        self.path = np.asarray(self.path)
        self.intercept = self.intercept - np.sum(beta * x_mu / x_sigma)
        self.path = self.path / x_sigma

        return 0

    def get_coef_path(self):
        return self.intercept, self.path
