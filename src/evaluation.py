import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from config import FIG_DIR

os.makedirs(FIG_DIR, exist_ok=True)


def metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    corr = float(np.corrcoef(np.ravel(y_true), np.ravel(y_pred))[0, 1])
    return {"MAE": mae, "MSE": mse, "Corr": corr}


def plot_train_val_test(all_y, split_ix, y_pred_test=None, name="model"):
    """График actual vs predicted по шаблону ЛР."""
    plt.figure(figsize=(12, 4))
    tr_ix, val_ix, te_ix = split_ix
    plt.plot(tr_ix, all_y[tr_ix], color="tab:blue", label="Actual Train")
    plt.plot(val_ix, all_y[val_ix], color="tab:green", label="Actual Validation")
    plt.plot(te_ix, all_y[te_ix], color="tab:purple", label="Actual Test")
    if y_pred_test is not None:
        plt.plot(te_ix, y_pred_test, color="tab:red", label="Predicted Test")
    plt.axvline(tr_ix[-1], color="gray", ls="--")
    plt.axvline(val_ix[-1], color="gray", ls="--")
    plt.legend(); plt.title(f"{name}: actual vs predicted")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/{name}_pred.png", dpi=150); plt.close()


def plot_diff(y_true, y_pred, name="model"):
    plt.figure(figsize=(10, 3))
    plt.plot(np.asarray(y_true) - np.asarray(y_pred))
    plt.title(f"{name}: residual (actual - predicted)")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/{name}_residual.png", dpi=150); plt.close()


def plot_tree_structure(tree_model, feature_names, name="decision_tree"):
    from sklearn.tree import plot_tree
    plt.figure(figsize=(16, 10))
    plot_tree(tree_model, feature_names=feature_names, filled=True, rounded=True)
    plt.savefig(f"{FIG_DIR}/{name}.png", dpi=120, bbox_inches="tight"); plt.close()


def plot_learning_curves(history, name="nn"):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.title("Loss"); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history["mae"], label="train")
    plt.plot(history.history["val_mae"], label="val")
    plt.title("MAE"); plt.legend()
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/{name}_curves.png", dpi=150); plt.close()


def plot_baseline_fit(x, y_true, y_pred, name="baseline"):
    order = np.argsort(x)
    plt.figure(figsize=(6, 4))
    plt.scatter(x, y_true, s=15, label="Actual")
    plt.plot(np.asarray(x)[order], np.asarray(y_pred)[order],
             "r--", label="Approximation")
    plt.legend(); plt.title(name)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/{name}.png", dpi=150); plt.close()


def results_table(rows, path):
    df = pd.DataFrame(rows)
    df.to_excel(path, index=False)
    return df