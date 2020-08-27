import numpy as np


class GradientDescent():

    def __init__(self, X=None, y=None,alpha=0.1,iter=100000):
        self.X = self.prepare_X(X)
        self.theta = np.ones((self.X.shape[1], 1)) * np.round(np.random.random(),2)
        self.y = self.transform_vector(y)
        self.alpha = alpha
        self.iter = iter
        self.J = self.objective_func(self.X, self.theta, self.y)

    def prepare_X(self, X):
        theta_X0 = np.ones((X.shape[0], 1))
        return np.concatenate((theta_X0, X), axis=1)

    def transform_vector(self, arr):
        arr = np.array(arr)
        return arr.reshape((-1, 1))

    def calculate_delta(self, X, theta, y):
        return (1 / X.shape[0] * (X.dot(theta) - y).reshape((1, -1)).dot(X)).reshape((-1, 1))

    def update_theta(self, theta, alpha, X, y):
        return theta - alpha * 1 / X.shape[0] * self.calculate_delta(X, theta, y)

    def objective_func(self, X, theta, y):
        return 1 / (2 * X.shape[0]) * (X.dot(theta) - y).reshape((1, -1)).dot((X.dot(theta) - y))

    def gradient_descent(self):
        for i in range(self.iter):
            self.theta = self.update_theta(self.theta, self.alpha, self.X, self.y)
            self.J = self.objective_func(self.X, self.theta, self.y)
            if self.J <= 0.0000001:
                return np.round(self.theta, 2)