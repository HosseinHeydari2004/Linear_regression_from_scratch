from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
from main import LinearRegression_scratch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def makeRegression():
    x, y = make_regression(
        n_samples=1000, n_features=2, n_targets=1, noise=3, random_state=42
    )
    return x, y

def virtuziation(x, y, title: str = "", xlabel: str = "", ylabel: str = ""):
    plt.figure(figsize=(20, 10), dpi=130)
    plt.scatter(x=x, y=y, s=150)
    plt.title(label=title, fontsize=20)
    plt.xlabel(xlabel=xlabel, fontsize=18)
    plt.ylabel(ylabel=ylabel, fontsize=18)
    plt.tick_params(axis="both", labelsize=17)
    plt.savefig()
    plt.show()


def run_model(x, y, name_model=LinearRegression):
    pass


def metrics():
    pass
