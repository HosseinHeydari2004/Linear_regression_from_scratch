import matplotlib
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from model import LinearRegression_scratch

matplotlib.use(backend="TkAgg")

x, y = make_regression(n_samples=1000, n_targets=1, n_features=2, noise=10, random_state=42)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)

model = LinearRegression_scratch()

model.fit(x_train, y_train.ravel(), True)

y_pred = model.predict(x_test)

print(f"MAE: {metrics.mean_absolute_error(y_test, y_pred)}")
print(f"MAE: {metrics.mean_squared_error(y_test, y_pred)}")
print(f"MAE: {metrics.root_mean_squared_error(y_test, y_pred)}")
print(f"MAE: {metrics.r2_score(y_test, y_pred)}")

sorted_idx = x_test[:, 0].argsort()
x_sorted = x_test[sorted_idx, 0]
y_test_sorted = y_test[sorted_idx]
y_pred_sorted = y_pred[sorted_idx]

plt.scatter(x_sorted, y_test_sorted, label="Actual")
plt.plot(x_sorted, y_pred_sorted, color="red", label="Predicted Line(model scratch)")
plt.xlabel("Feature 1")
plt.ylabel("Target")
plt.legend()
plt.show()
