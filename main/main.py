import numpy as np
import pandas as pd


class LinearRegression_scratch:
    """ """

    def __init__(self):
        self.w = None
        self.b = None

    def predict(self, X):
        """
        Y = Xw + b
        """
        return X @ self.w + self.b

    def _compute_loss(self, X, Y):
        """Mean Squared error loss\n
        MSE = 1/n(sum(Y_real - y_pred))
        """
        Y_pred = self.predict(X)
        Errors = Y - Y_pred
        loss = np.mean(Errors**2)
        return loss
