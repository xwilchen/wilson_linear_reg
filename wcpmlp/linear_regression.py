from wcpmlp.gradient_descent import GradientDescent
import numpy as np

class LinearReression():

    def __init__(self):
        self.theta = None

    def fit(self,X,y,alpha=0.1,iter=100000):
        gd = GradientDescent(X,y,alpha,iter)
        self.theta = gd.gradient_descent()

    def predict(self,X):
        theta_X0 = np.ones((X.shape[0], 1))
        X = np.concatenate((theta_X0, X), axis=1)
        pred = X.dot(self.theta)
        return pred