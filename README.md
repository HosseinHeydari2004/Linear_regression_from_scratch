# 📈 Linear Regression From Scratch

This project is a **complete from-scratch implementation of Linear Regression** using pure Python and NumPy, without relying on machine learning libraries such as `scikit-learn`.

The main goal is to deeply understand the **mathematics**, **optimization process**, and **learning behavior** behind linear regression and gradient descent.

---

## 🧠 Concepts Covered

- Simple & Multiple Linear Regression  
- Hypothesis Function  
- Mean Squared Error (MSE)  
- Gradient Descent Algorithm  
- Weight & Bias Updates  
- Learning Rate Effects  
- Model Convergence  
- Overfitting & Underfitting (intuition-based understanding)  

---

## 🛠️ Technologies Used

- Python 🐍  
- NumPy  
- Matplotlib  

---

## 🧮 Mathematical Background

### Hypothesis Function
ŷ = Xw + b

### Mean Squared Error (Loss Function)

MSE = (1 / n) * Σ (yᵢ − ŷᵢ)²

### Gradients
∂L / ∂w = (2 / n) · Xᵀ(ŷ − y)<br>
∂L / ∂b = (2 / n) · Σ(ŷ − y)

---

## ⚙️ Linear Regression Implementation (From Scratch)

```python

class LinearRegressionScratch:
    """
    Linear Regression implemented from scratch using Gradient Descent
    """

    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
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

    def _compute_loss(self, X: np.ndarray, y: np.ndarray):
        """
        Mean Squared error loss\n
        MSE = 1/n(sum(Y_real - y_pred))
        """
        y_pred = self.predict(X)
        return np.mean((y - y_pred) ** 2)

    def _compute_gradients(self, X: np.ndarray, y: np.ndarray):
        """
        Compute gradients of loss w.r.t weights and bias
        """
        n_samples = X.shape[0]
        y_pred = self.predict(X)

        error = y_pred - y

        dw = (2 / n_samples) * (X.T @ error)
        db = (2 / n_samples) * np.sum(error)

        return dw, db

    def fit(self, X: np.ndarray, y: np.ndarray, verbose=False):
        """
        Train the model using Gradient Descent
        """
        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0.0

        for i in range(self.n_iters):
            dw, db = self._compute_gradients(X, y)

            # Update parameters
            self.w -= self.lr * dw
            self.b -= self.lr * db

            # Track loss
            loss = self._compute_loss(X, y)
            self.loss_history.append(loss)

            if verbose and i % 100 == 0:
                print(f"Iteration {i}: Loss = {loss:.6f}")

        return self
```

---

## 📊 Observations
- Proper learning rate leads to smooth convergence
- Large learning rates may cause divergence
- Small learning rates slow down training
- Gradient Descent successfully finds optimal parameters

---

## 🎯 Key Takeaways
- Linear Regression is an optimization problem
- Gradient Descent minimizes loss iteratively
- Understanding math improves debugging and intuition
- Implementing from scratch strengthens ML fundamentals

---

## 🚀 Future Improvements

- Add L1 / L2 Regularization
- Implement Normal Equation
- Add R² Score
- Mini-batch & Stochastic Gradient Descent
- Compare with sklearn.LinearRegression

---

## 👤 Author

Hossein Heydari
Computer Engineering Student | Junior Data Scientist and Machine Learning Engineer

GitHub:<br>
Kaggle:

---

## 🤝 Contributing

Contributions are welcome and appreciated 🙌

If you have ideas for improvements, bug fixes, or new features:

- Fork the repository

- Create a new branch

- Commit your changes

- Open a Pull Request

All contributions help make this project better and more educational.

---

## ❤️ Support

If you find this project helpful, please ⭐ the repository and share your feedback.

Happy Learning 🌱✨

