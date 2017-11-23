from __future__ import print_function
from sklearn.preprocessing import StandardScaler
import numpy as np

def get_rating_error(r, p, q):
    return r - np.dot(p, q)

def get_error(R, P, Q, beta):
    error = 0.0
    for i in xrange(len(R)):
        for j in xrange(len(R[i])):
            if R[i][j] == 0:
                continue
            error += pow(get_rating_error(R[i][j], P[:,i], Q[:,j]), 2)
    error += beta/2.0 * (np.linalg.norm(P) + np.linalg.norm(Q))
    return error

def matrix_factorization(R, K, steps=5000, alpha=0.0002, beta=0.02, threshold=0.001):
    P = np.random.rand(K, len(R))
    Q = np.random.rand(K, len(R[0]))
    for step in xrange(steps):
        for i in xrange(len(R)):
            for j in xrange(len(R[i])):
                if R[i][j] == 0:
                    continue
                err = get_rating_error(R[i][j], P[:, i], Q[:, j])
                for k in xrange(K):
                    P[k][i] += alpha * (2 * err * Q[k][j])
                    Q[k][j] += alpha * (2 * err * P[k][i])
        error = get_error(R, P, Q, beta)
        if error < threshold:
            break
    return P, Q
    
class ReducedRankRegression:
    def __init__(self, n_components):
        self.n_components = n_components
        
    def fit(self, sample_X, sample_Y):
        sample_X = np.array(sample_X)
        sample_Y = np.array(sample_Y)
        
        self.standard_scaler_X = StandardScaler()
        normalized_X = self.standard_scaler_X.fit_transform(sample_X)
        self.standard_scaler_Y = StandardScaler(with_std=False)
        normalized_Y = self.standard_scaler_Y.fit_transform(sample_Y)
        
        cov_matrix = np.zeros((normalized_Y.shape[1], normalized_X.shape[1]))
        for vec_y, vec_x in zip(normalized_Y, normalized_X):
            cov_matrix += np.dot(vec_y.reshape(-1, 1), vec_x.reshape(1, -1))
        cov_matrix /= sample_X.shape[0]
        
        self.W, self.H = matrix_factorization(cov_matrix, self.n_components)
    
    def fit_transform(self, sample_X, sample_Y):
        self.fit(sample_X, sample_Y)
        return self.H
    
    def transform(self, sample_X):
        normalized_X = self.standard_scaler_X.transform(sample_X)
        return np.dot(normalized_X, self.H.T)

    def predict(self, sample_X):
        normalized_X = self.standard_scaler_X.transform(sample_X)
        return np.dot(self.W, np.dot(self.H, sample_X.T)).T + self.standard_scaler_Y.mean_.reshape(1, -1)



if __name__ == "__main__":
    from sklearn.metrics import mean_squared_error
    import matplotlib.pyplot as plt
    def generateSample(n_sample):
        true_W = np.array([[1., 2.], [3., 4.]])
        true_H = true_W.T
        sample_X = np.random.normal(size=(n_sample, 2))
        sample_Y = np.dot(true_W, np.dot(true_H, sample_X.T)).T + np.random.normal(n_sample)
        return sample_X, sample_Y
    
    sample_X, sample_Y = generateSample(1000)
    reduced_rank_regression = ReducedRankRegression(2)
    reduced_rank_regression.fit(sample_X, sample_Y)
    est_Y = reduced_rank_regression.predict(sample_X)
    plt.scatter(sample_Y[:, 0], est_Y[:, 0], c='r')
    plt.scatter(sample_Y[:, 1], est_Y[:, 1], c='g')
    plt.show()
    print(mean_squared_error(sample_Y[:, 0], est_Y[:, 0]))
    print(mean_squared_error(sample_Y[:, 1], est_Y[:, 1]))

