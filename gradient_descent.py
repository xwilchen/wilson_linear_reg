import numpy as np

class Gradient_Descent():
    def __init__(self,alpha=0.1,theta=None,X=None,y=None,iteration=1000):
        self.alpha = alpha
        self.theta = self.transform_vector(theta)
        self.X = self.prepare_X(X)
        self.y = self.transform_vector(y)
        self.iteration = iteration
        self.J = None

    def prepare_X(self,X):
        theta_X0 = np.ones((X.shape[0],1))
        return np.concatenate((theta_X0,X),axis=1)

    def transform_vector(self,arr):
        if arr.shape[1] != 1:
            return arr.T
        return arr

    def calculate_delta(self,X,theta,y):
        return 1/X.shape[0] * (X.dot(theta) - y).T.dot(X)

    def update_theta(self,theta,alpha,X,y):
        return theta - alpha * 1/len(y) * self.calculate_delta(X,theta,y)

    def objective_func(self,X,theta,y):
        return 1/(2*len(y)) * (X.dot(theta) - y).T.dot((X.dot(theta) - y))

    def gradient_descent(self):
        for i in range(self.iteration):
            previous_theta = self.theta
            self.theta = self.update_theta(self.theta,self.alpha,self.X,self.y)
            self.J = self.objective_func(self.X,self.theta,self.y)
            if abs(previous_theta - self.theta) < 0.0001:
                return (self.theta,self.J)