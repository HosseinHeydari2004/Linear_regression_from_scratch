import numpy as np


class LinearRegression_scratch:
    """Linear Regression implemented from scratch using Gradient Descent"""

    def __init__(self, n_iters=1000):
        self.n_iters = n_iters
        self.w = None
        self.b = None
        self.loss_history = []

    def predict(self, X: np.ndarray):
        """
        Y = Xw + b\n
        Here:\n
        X = Matrix of features\n
        w = Weights vector\n
        b = intercept
        """
        return X @ self.w + self.b

    def _compute_loss(self, X: np.ndarray, Y: np.ndarray):
        """Mean Squared error loss\n
        MSE = 1/n(sum(Y_real - y_pred))
        """
        Y_pred = self.predict(X)
        Errors = Y - Y_pred
        loss = np.mean(Errors ** 2)
        return loss

    def _compute_gradients(self, X: np.ndarray, Y: np.ndarray):
        """Compute gradients of loss w.r.t weights and bias"""
        n_samples = X.shape[0]
        y_pred = self.predict(X)
        Errors = y_pred - Y

        dw = (2 / n_samples) * (X.T @ Errors)
        db = (2 / n_samples) * np.sum(Errors)
        return dw, db

    def fit(self, X: np.ndarray, Y:np.ndarray, Verbose=False):
        """Train the model using Gradient Descent"""
        n_samples, n_feature = X.shape
        self.w = np.zeros(n_feature)
        self.b = 0.0
        lr0 = 0.1
        decay = 0.01
        for i in range(self.n_iters):
            lr = lr0 / (1 + decay * i)
            dw, db = self._compute_gradients(X, Y)
            # Update parameters
            self.w -= lr * dw
            self.b -= lr * db
            loss = self._compute_loss(X, Y)
            self.loss_history.append(loss)

            if Verbose and i % 100 == 0:
                print(f"iter {i}: loss={loss:.6f}")
        return self
