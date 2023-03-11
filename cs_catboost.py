
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import time

df = pd.read_csv("Concrete_Data_yeh.csv")

X = df.drop("csMPa", axis=1)
y = df["csMPa"]

# Final cross validation scoring with best parameters
print("""Best Hyperparameters for CatBoostRegressor: 
    iterations: 10,000
    depth: 7
    learning_rate: 0.2, 
    l2_leaf_reg: 3 """)
start = time.time()

regressor = CatBoostRegressor(iterations=10000, depth=7, learning_rate=0.2, l2_leaf_reg=3, random_state=42)
scores = cross_val_score(regressor, X, y, cv=5, scoring="neg_root_mean_squared_error")
print(f"5-Fold cross validation scores: \n\t{-scores}")
print(f"Average score: \n\t{-scores.mean()}")

end = time.time()
print(f"Time to run: {end-start}")


# Preliminary random search for good parameters

# regressor = CatBoostRegressor(random_state=42)
# params = {
#     "iterations": [500],
#     "depth": randint(6, 8),
#     "learning_rate": [0.00025 * i for i in range(0, 1000)],
#     "l2_leaf_reg": randint(1, 20)
# }
# rs = RandomizedSearchCV(regressor, params, cv=5, scoring="neg_root_mean_squared_error")
# rs.fit(X, y)
#
# print(f"Best params: {rs.best_params_}")
# print(f"Best score: {-rs.best_score_}")


# Secondary grid search for parameters

# regressor = CatBoostRegressor(random_state=42)
# grid_params = {
#     "iterations": [500],
#     "depth": [6, 7, 8],
#     "learning_rate": [0.01, 0.10, 0.15, 0.2],
#     "l2_leaf_reg": [2, 3, 4, 5]
# }
# gs = GridSearchCV(regressor, grid_params, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
# gs.fit(X, y)
#
# print(f"Best params: {gs.best_params_}")
# print(f"Best score: {-gs.best_score_}")

