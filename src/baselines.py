import numpy as np
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

def single_linear_regression(X_series: np.ndarray, y_series: np.ndarray):
    """
    Парная линейная регрессия y ~ x по ОДНОМУ признаку.
    X_series: (N,) или (N,1).
    """
    x = np.asarray(X_series).reshape(-1, 1)
    model = LinearRegression().fit(x, y_series)
    return model, lambda xnew: model.predict(np.asarray(xnew).reshape(-1, 1))


def multiple_linear_regression(X: np.ndarray, y: np.ndarray):
    model = LinearRegression().fit(X, y)
    return model, model.predict


# Пример нелинейной функции: a * sin(b*x) + c*x + d
def nonlinear_func(X, a, b, c, d):
    x1 = X[:, 0]
    x2 = X[:, 1] if X.ndim > 1 and X.shape[1] > 1 else np.zeros_like(x1)
    return a * np.sin(b * x1) + c * x2 + d


def nonlinear_regression(X: np.ndarray, y: np.ndarray,
                         func=nonlinear_func, p0=None):
    if p0 is None:
        p0 = [1.0, 0.01, 0.1, float(np.mean(y))]
    popt, _ = curve_fit(func, X, y, p0=p0, maxfev=20000)
    pred = lambda Xn: func(Xn, *popt)
    return popt, pred