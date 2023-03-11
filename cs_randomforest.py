import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

df = pd.read_csv("Concrete_Data_Yeh.csv")
X = df.drop("csMPa", axis=1)
y = df["csMPa"].copy()

print("""Best Hyperparameters for CatBoostRegressor: 
    n_estimators: 1400
    max_depth: 24
    min_samples_leaf: 1, 
    min_samples_split: 2
    max_features: 'sqrt' """)

regressor = RandomForestRegressor(n_estimators=1400, max_depth=24, min_samples_leaf=1, min_samples_split=2,
                                  max_features='sqrt', random_state=42)

scores = -cross_val_score(regressor, X, y, cv=5, scoring="neg_root_mean_squared_error")
print(f"Scores: {scores}")
print(f"Mean Score: {scores.mean()}")


# Initial random parameter search to determine which parameters are significant in which direction

# params = {
#     "n_estimators": [i * 200 for i in range(0, 15)],
#     "max_depth": randint(low=0, high=50),
#     "min_samples_split": randint(low=1, high=20),
#     "max_features": ["sqrt"],
#     "min_samples_leaf": randint(low=1, high=15)
# }
# cv = RandomizedSearchCV(regressor, params, cv=5, scoring="neg_root_mean_squared_error", random_state=42)
# cv.fit(X, y)
# print(f"Best Params: {cv.best_params_}")
# print(f"Best Score: {-cv.best_score_}")

# Second grid search to fine tune parameters more

# regressor = RandomForestRegressor(random_state=42)
# grid_params = {
#     "n_estimators": [1400],
#     "max_depth": [18, 19, 20, 21, 22, 23, 24],
#     "min_samples_leaf": [1, 2, 3, 4],
#     "min_samples_split": [2, 4, 6],
#     "max_features": ["sqrt"]
# }
# gs = GridSearchCV(regressor, grid_params, cv=5, scoring="neg_root_mean_squared_error")
# gs.fit(X, y)
#
# print(f"Best Params: {gs.best_params_}")
# print(f"Best Score: {-gs.best_score_}")
#
# '''