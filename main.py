"""
Главный скрипт лабораторной работы.
Запускать из корня проекта: python main.py
"""

import matplotlib
matplotlib.use('Agg')  # Отключает вывод окон, сохраняет только в файлы
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (LAGS, STEP, MULTI_HORIZON, RESULTS_PATH,
                    RANDOM_STATE, VARIANT)
from src.data_loader import (load_dataset, make_lagged_features,
                             make_multi_horizon_target)
from src.preprocessing import (split_series, fit_scaler_on_train,
                               apply_scaler, correlation_matrix,
                               best_corr_feature)
from src import baselines, single_models, ensemble_models, neural_models
from src import evaluation

np.random.seed(RANDOM_STATE)

def main():
    # ------------------------------------------------------------------
    # 1. Данные
    # ------------------------------------------------------------------
    print("== Загрузка датасета ==")
    df = load_dataset(variant=VARIANT)
    print(df.head())

    # 1.1 График всех признаков и y
    import matplotlib.pyplot as plt, os
    os.makedirs("outputs/figures", exist_ok=True)
    plt.figure(figsize=(14, 8))
    for c in df.columns:
        plt.subplot(len(df.columns), 1, df.columns.get_loc(c)+1)
        plt.plot(df[c].values, lw=0.8)
        plt.ylabel(c)
    plt.tight_layout()
    plt.savefig("outputs/figures/all_features.png", dpi=120)
    plt.close()

    # 1.3 Матрица корреляций
    corr = correlation_matrix(df)
    print("\n== Матрица корреляций ==\n", corr)
    best_feat = best_corr_feature(df, "y")
    print(f"Признак с наибольшей |corr| с y: {best_feat}")

    # Лаги (это X, y — сдвинутая y)
    X, y = make_lagged_features(df, lags=LAGS, target="y")
    print("Форма X, y после лагов:", X.shape, y.shape)

    # 1.2 Разбиение и масштабирование (scaler обучен на train)
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = split_series(X, y)
    scaler = fit_scaler_on_train(X_tr)
    X_tr_s = apply_scaler(scaler, X_tr)
    X_val_s = apply_scaler(scaler, X_val)
    X_te_s = apply_scaler(scaler, X_te)

    split_ix = (np.array(X_tr.index), np.array(X_val.index), np.array(X_te.index))
    all_y = y.values

    results = []   # таблица метрик

    # ------------------------------------------------------------------
    # 1.4 Парная линейная регрессия
    # ------------------------------------------------------------------
    split_ix = (np.array(X_tr.index), np.array(X_val.index), np.array(X_te.index))
    all_y = y.values

    results = []   # таблица метрик

    # ------------------------------------------------------------------
    # 1.4 Парная линейная регрессия
    # ------------------------------------------------------------------
    # Безопасно получаем значения признака по индексам из X_tr, X_val, X_te
    x_best_tr = X_tr[best_feat + "_lag0"].values
    x_best_val = X_val[best_feat + "_lag0"].values
    x_best_te = X_te[best_feat + "_lag0"].values

    model_pair, f_pair = baselines.single_linear_regression(x_best_tr, y_tr.values)
    yp_pair_te = f_pair(x_best_te)
    
    results.append({"model": "Pair Linear", **evaluation.metrics(y_te.values, yp_pair_te)})
    evaluation.plot_baseline_fit(x_best_te, y_te.values, yp_pair_te, name="pair_linear")
    evaluation.plot_train_val_test(all_y, split_ix, yp_pair_te, name="pair_linear")

    # ------------------------------------------------------------------
    # 1.5 Множественная линейная регрессия
    # ------------------------------------------------------------------
    model_multi, f_multi = baselines.multiple_linear_regression(X_tr_s, y_tr.values)
    yp_multi_te = f_multi(X_te_s)
    results.append({"model": "Multiple Linear", **evaluation.metrics(y_te.values, yp_multi_te)})
    evaluation.plot_train_val_test(all_y, split_ix, yp_multi_te, name="multiple_linear")

    # ------------------------------------------------------------------
    # 1.6 Нелинейная регрессия (curve_fit)
    # ------------------------------------------------------------------
    popt, f_nonlin = baselines.nonlinear_regression(X_tr_s, y_tr.values)
    yp_nonlin_te = f_nonlin(X_te_s)
    results.append({"model": "Nonlinear (curve_fit)", **evaluation.metrics(y_te.values, yp_nonlin_te)})
    evaluation.plot_train_val_test(all_y, split_ix, yp_nonlin_te, name="nonlinear")

    # ------------------------------------------------------------------
    # 2. Одиночные методы
    # ------------------------------------------------------------------
    print("\n== KNN ==")
    knn = single_models.knn_regression(X_tr_s, y_tr.values, X_val_s, y_val.values)
    yp_knn = knn.predict(X_te_s)
    results.append({"model": "KNN", **evaluation.metrics(y_te.values, yp_knn)})
    evaluation.plot_train_val_test(all_y, split_ix, yp_knn, name="knn")

    print("== SVM ==")
    svm = single_models.svm_regression(X_tr_s, y_tr.values)
    yp_svm = svm.predict(X_te_s)
    results.append({"model": "SVM", **evaluation.metrics(y_te.values, yp_svm)})
    evaluation.plot_train_val_test(all_y, split_ix, yp_svm, name="svm")

    print("== Decision Tree ==")
    dt = single_models.decision_tree_regression(X_tr_s, y_tr.values)
    yp_dt = dt.predict(X_te_s)
    results.append({"model": "Decision Tree", **evaluation.metrics(y_te.values, yp_dt)})
    evaluation.plot_tree_structure(dt, X.columns, name="decision_tree")
    evaluation.plot_train_val_test(all_y, split_ix, yp_dt, name="decision_tree")

    # ------------------------------------------------------------------
    # 3. Ансамбли
    # ------------------------------------------------------------------
    print("\n== Random Forest ==")
    rf = ensemble_models.random_forest(X_tr_s, y_tr.values)
    yp_rf = rf.predict(X_te_s)
    results.append({"model": "Random Forest", **evaluation.metrics(y_te.values, yp_rf)})
    evaluation.plot_train_val_test(all_y, split_ix, yp_rf, name="random_forest")

    print("== Bagging ==")
    bg = ensemble_models.bagging(X_tr_s, y_tr.values)
    yp_bg = bg.predict(X_te_s)
    results.append({"model": "Bagging", **evaluation.metrics(y_te.values, yp_bg)})
    evaluation.plot_train_val_test(all_y, split_ix, yp_bg, name="bagging")

    print("== Gradient Boosting ==")
    gb = ensemble_models.gradient_boosting(X_tr_s, y_tr.values)
    yp_gb = gb.predict(X_te_s)
    results.append({"model": "Gradient Boosting", **evaluation.metrics(y_te.values, yp_gb)})
    evaluation.plot_train_val_test(all_y, split_ix, yp_gb, name="gradient_boosting")

    # Опционально XGBoost
    try:
        print("== XGBoost ==")
        xgb = ensemble_models.xgboost_regression(X_tr_s, y_tr.values)
        yp_xgb = xgb.predict(X_te_s)
        results.append({"model": "XGBoost", **evaluation.metrics(y_te.values, yp_xgb)})
        evaluation.plot_train_val_test(all_y, split_ix, yp_xgb, name="xgboost")
    except Exception as e:
        print("XGBoost пропущен:", e)

    # ------------------------------------------------------------------
    # 4. Нейронные сети (1 шаг и 7 шагов)
    # ------------------------------------------------------------------
    n_features = X_tr_s.shape[1]

    # 4.1 Однослойный персептрон
    print("\n== Single-layer perceptron ==")
    m1 = neural_models.build_single_layer(n_features, activation="relu")
    h1 = neural_models.train_model(m1, X_tr_s, y_tr.values, X_val_s, y_val.values,
                                   epochs=100, batch_size=32)
    yp_nn1 = m1.predict(X_te_s).ravel()
    results.append({"model": "1-layer Perceptron",
                    **evaluation.metrics(y_te.values, yp_nn1),
                    "params": neural_models.count_params(m1)})
    evaluation.plot_learning_curves(h1, name="single_layer_nn")
    evaluation.plot_train_val_test(all_y, split_ix, yp_nn1, name="single_layer_nn")

    # 4.2 Многослойный персептрон
    print("== MLP ==")
    m2 = neural_models.build_mlp(n_features, hidden=(64, 32))
    h2 = neural_models.train_model(m2, X_tr_s, y_tr.values, X_val_s, y_val.values,
                                   epochs=120, batch_size=32)
    yp_nn2 = m2.predict(X_te_s).ravel()
    results.append({"model": "MLP",
                    **evaluation.metrics(y_te.values, yp_nn2),
                    "params": neural_models.count_params(m2)})
    evaluation.plot_learning_curves(h2, name="mlp")
    evaluation.plot_train_val_test(all_y, split_ix, yp_nn2, name="mlp")

    # 4.3 Глубокая сеть
    print("== Deep NN ==")
    m3 = neural_models.build_deep(n_features)
    h3 = neural_models.train_model(m3, X_tr_s, y_tr.values, X_val_s, y_val.values,
                                   epochs=150, batch_size=32)
    yp_nn3 = m3.predict(X_te_s).ravel()
    results.append({"model": "Deep NN",
                    **evaluation.metrics(y_te.values, yp_nn3),
                    "params": neural_models.count_params(m3)})
    evaluation.plot_learning_curves(h3, name="deep_nn")
    evaluation.plot_train_val_test(all_y, split_ix, yp_nn3, name="deep_nn")

    # 4. Многошаговый прогноз (7 шагов) — на MLP
    print("\n== Multi-horizon (7 steps) ==")
    Xm, Ym = make_multi_horizon_target(df, lags=LAGS, horizon=MULTI_HORIZON, target="y")
    (Xm_tr, Ym_tr), (Xm_val, Ym_val), (Xm_te, Ym_te) = split_series(Xm, Ym)

    sc_m = fit_scaler_on_train(Xm_tr)
    Xm_tr_s = apply_scaler(sc_m, Xm_tr)
    Xm_val_s = apply_scaler(sc_m, Xm_val)
    Xm_te_s = apply_scaler(sc_m, Xm_te)

    mh = neural_models.build_multi_horizon_mlp(Xm_tr_s.shape[1],
                                               horizon=MULTI_HORIZON)
    hh = neural_models.train_model(mh, Xm_tr_s, Ym_tr.values,
                                   Xm_val_s, Ym_val.values,
                                   epochs=150, batch_size=32)
    Ym_pred = mh.predict(Xm_te_s)

    # усреднённые метрики по 7 горизонтам
    mae_h = float(np.mean(np.abs(Ym_pred - Ym_te.values)))
    mse_h = float(np.mean((Ym_pred - Ym_te.values) ** 2))
    corr_h = float(np.corrcoef(Ym_pred.ravel(), Ym_te.values.ravel())[0, 1])
    results.append({"model": "MLP multi-horizon (7)",
                    "MAE": mae_h, "MSE": mse_h, "Corr": corr_h,
                    "params": neural_models.count_params(mh)})
    evaluation.plot_learning_curves(hh, name="multi_horizon_mlp")

    # ------------------------------------------------------------------
    # 5.5 Итоговая таблица
    # ------------------------------------------------------------------
    df_res = evaluation.results_table(results, RESULTS_PATH)
    print("\n== Итоговые метрики ==")
    print(df_res)


if __name__ == "__main__":
    main()