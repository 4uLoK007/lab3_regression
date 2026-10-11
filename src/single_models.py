from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

def knn_regression(X_tr, y_tr, X_val, y_val, param_grid=None):
    param_grid = param_grid or {
        "n_neighbors": [2, 3, 5, 7, 10, 15],
        "weights": ["uniform", "distance"]
    }
    gs = GridSearchCV(
        KNeighborsRegressor(), 
        param_grid, 
        cv=3,
        scoring="neg_mean_squared_error", 
        n_jobs=-1
    )
    gs.fit(X_tr, y_tr)
    return gs.best_estimator_


def svm_regression(X_tr, y_tr, param_grid=None):
    param_grid = param_grid or {
        "C": [0.1, 1, 10],
        "epsilon": [0.01, 0.1, 0.5],
        "kernel": ["rbf"]
    }
    gs = GridSearchCV(
        SVR(), 
        param_grid, 
        cv=3,
        scoring="neg_mean_squared_error", 
        n_jobs=-1
    )
    gs.fit(X_tr, y_tr)
    return gs.best_estimator_


def decision_tree_regression(X_tr, y_tr, param_grid=None):
    param_grid = param_grid or {
        "max_depth": [3, 5, 8, 12, None],
        "min_samples_leaf": [1, 2, 5]
    }
    gs = GridSearchCV(
        DecisionTreeRegressor(random_state=0), 
        param_grid,
        cv=3, 
        scoring="neg_mean_squared_error", 
        n_jobs=-1
    )
    gs.fit(X_tr, y_tr)
    return gs.best_estimator_