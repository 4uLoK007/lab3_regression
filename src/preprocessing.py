import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from config import TEST_SIZE, VAL_SIZE, RANDOM_STATE

def split_series(X: pd.DataFrame, y: pd.Series,
                 test_size=TEST_SIZE, val_size=VAL_SIZE):
    """
    Разбиение по времени (без перемешивания!):
      train -> val -> test
    Возвращает словари {split: (X, y)} и индексы.
    """
    n = len(X)
    n_test = int(n * test_size)
    n_val = int(n * val_size)
    n_train = n - n_test - n_val

    X_tr = X.iloc[:n_train]
    y_tr = y.iloc[:n_train]

    X_val = X.iloc[n_train:n_train + n_val]
    y_val = y.iloc[n_train:n_train + n_val]

    X_te = X.iloc[n_train + n_val:]
    y_te = y.iloc[n_train + n_val:]

    return (X_tr, y_tr), (X_val, y_val), (X_te, y_te)


def fit_scaler_on_train(X_tr: pd.DataFrame):
    scaler = StandardScaler()
    scaler.fit(X_tr)
    return scaler


def apply_scaler(scaler, X: pd.DataFrame) -> np.ndarray:
    return scaler.transform(X)


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df.corr(numeric_only=True)


def best_corr_feature(df: pd.DataFrame, target="y") -> str:
    corr = df.corr(numeric_only=True)[target].drop(target).abs()
    return corr.idxmax()