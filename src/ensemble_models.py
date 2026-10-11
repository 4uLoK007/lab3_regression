from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

def random_forest(X_tr, y_tr, param_grid=None):
    param_grid = param_grid or {"n_estimators": [100, 200],
                                "max_depth": [5, 10, None],
                                "min_samples_leaf": [1, 2, 5]}
    gs = GridSearchCV(RandomForestRegressor(random_state=0), param_grid,
                      cv=3, scoring="neg_mean_squared_error", n_jobs=-1)
    gs.fit(X_tr, y_tr)
    return gs.best_estimator_


def bagging(X_tr, y_tr, param_grid=None):
    param_grid = param_grid or {"n_estimators": [20, 50, 100],
                                "estimator__max_depth": [5, 10, None]}
    gs = GridSearchCV(BaggingRegressor(estimator=DecisionTreeRegressor(),
                                       random_state=0),
                      param_grid, cv=3, scoring="neg_mean_squared_error",
                      n_jobs=-1)
    gs.fit(X_tr, y_tr)
    return gs.best_estimator_


def gradient_boosting(X_tr, y_tr, param_grid=None):
    param_grid = param_grid or {"n_estimators": [100, 200],
                                "learning_rate": [0.05, 0.1],
                                "max_depth": [3, 5]}
    gs = GridSearchCV(GradientBoostingRegressor(random_state=0), param_grid,
                      cv=3, scoring="neg_mean_squared_error", n_jobs=-1)
    gs.fit(X_tr, y_tr)
    return gs.best_estimator_


# опционально XGBoost
def xgboost_regression(X_tr, y_tr, param_grid=None):
    from xgboost import XGBRegressor
    param_grid = param_grid or {"n_estimators": [200, 400],
                                "max_depth": [3, 5, 7],
                                "learning_rate": [0.05, 0.1]}
    gs = GridSearchCV(XGBRegressor(objective="reg:squarederror",
                                   random_state=0), param_grid,
                      cv=3, scoring="neg_mean_squared_error", n_jobs=-1)
    gs.fit(X_tr, y_tr)
    return gs.best_estimator_