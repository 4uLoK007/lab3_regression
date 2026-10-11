import pandas as pd
import numpy as np
from config import DATASET_PATH, VARIANT

def _find_variant_block(header_row, variant: int):
    """Находит столбцы нужного варианта в строке заголовков."""
    label = f"Вариант {variant}"
    try:
        start = header_row.index(label)
    except ValueError:
        raise ValueError(f"Не найден столбец {label}")
    # у каждого варианта 6 столбцов: a,b,c,d,e,y
    cols = list(range(start, start + 6))
    return cols

def load_dataset(path: str = DATASET_PATH, variant: int = VARIANT) -> pd.DataFrame:
    """
    Загружает лист Excel с 'двухэтажным' заголовком:
      строка 1 — 'Вариант N'
      строка 2 — 'a,b,c,d,e,y'
    Возвращает DataFrame с колонками a,b,c,d,e,y для выбранного варианта.
    """
    raw = pd.read_excel(path, header=None)

    # строка с "Вариант N" — обычно 0-я
    header_row = list(raw.iloc[0].values)
    cols = _find_variant_block(header_row, variant)

    names_row = raw.iloc[1].values
    col_names = [names_row[i] for i in cols]

    df = raw.iloc[2:, cols].copy()
    df.columns = col_names
    df = df.apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
    return df


def make_lagged_features(df: pd.DataFrame, lags: int = 5, target: str = "y"):
    """
    Формирует таблицу с лагами всех признаков (включая y):
      X = [a_{t}, a_{t-1},..., a_{t-lags+1}, ..., y_{t-1},...]
      target = y_{t}
    """
    feat_cols = [c for c in df.columns if c != target]
    frames = []
    for c in feat_cols + [target]:
        for lag in range(lags):
            frames.append(df[c].shift(lag).rename(f"{c}_lag{lag}"))
    X = pd.concat(frames, axis=1)
    y = df[target].rename("y_target")

    data = pd.concat([X, y], axis=1).dropna().reset_index(drop=True)
    X = data.drop(columns=["y_target"])
    y = data["y_target"]
    return X, y


def make_multi_horizon_target(df: pd.DataFrame, lags: int, horizon: int = 7, target="y"):
    """
    Многошаговый прогноз: таргет — вектор [y_{t+1},...,y_{t+horizon}].
    Признаки — лаги всех столбцов на момент t.
    """
    feat_cols = [c for c in df.columns if c != target]
    frames = []
    for c in feat_cols + [target]:
        for lag in range(lags):
            frames.append(df[c].shift(lag).rename(f"{c}_lag{lag}"))
    X = pd.concat(frames, axis=1)

    Y = pd.concat([df[target].shift(-h) for h in range(1, horizon + 1)], axis=1)
    Y.columns = [f"y_t+{h}" for h in range(1, horizon + 1)]

    data = pd.concat([X, Y], axis=1).dropna().reset_index(drop=True)
    return data.drop(columns=Y.columns), data[Y.columns]