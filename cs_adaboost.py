import pandas as pd
from sklearn.ensemble import AdaBoostRegressor
import time

from sklearn.model_selection import cross_val_score, RandomizedSearchCV, GridSearchCV

df = pd.read_csv("Concrete_Data_Yeh.csv")
X = df.drop("csMPa", axis=1)
y = df["csMPa"].copy()
#
print("""Adaboost Regression -- Best Hyperparameters:
    n_estimators: 100,
    learning_rate: 1.7
    """)

regressor = AdaBoostRegressor(n_estimators=100, learning_rate=1.7)
scores = -cross_val_score(regressor, X=X, y=y, cv=5, scoring="neg_root_mean_squared_error")

print(f"5-fold cross validation scores: \n\t{scores}")
print(f"Average score: \n\t{scores.mean()}")



# Initial Random search for parameters

# regressor = AdaBoostRegressor()
#
# random_params = {
#     "n_estimators": [1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000],
#     "learning_rate": [0.0025 * i for i in range(0, 1000)]
# }
# start = time.time()
# cv = RandomizedSearchCV(regressor, random_params, cv=5, scoring="neg_root_mean_squared_error", random_state=42,
#                         n_iter=25)
# cv.fit(X, y)
# print(f"Best Params: {cv.best_params_}")
# print(f"Best Score: {-cv.best_score_}")
# end = time.time()
# print(f"Time elapsed: {end - start}")
#
# # Code for displaying top 10 random search results
# results = pd.DataFrame(cv.cv_results_)
# results.sort_values(by="mean_test_score", ascending=False, inplace=True)
# results = results.drop(
#     ["mean_fit_time", "std_fit_time", "mean_score_time", "std_score_time", "std_test_score", "params"], axis=1)
# pd.set_option('display.max_columns', None)
# print(f"Results: {results.head(10)}")

####  Final grid search

# regressor = AdaBoostRegressor()
# params = {
#     "n_estimators": [50, 100, 200, 500],
#     "learning_rate": [0.01 * i for i in range(0, 250)]
# }
#
# rs = GridSearchCV(regressor, params, cv=5, scoring="neg_root_mean_squared_error", n_jobs=-1)
# rs.fit(X, y)
# print(f"Best Params: {rs.best_params_}")
# print(f"Best Score: {-rs.best_score_}")
# # Code for displaying top 10 random search results
# results = pd.DataFrame(rs.cv_results_)
# results.sort_values(by="mean_test_score", ascending=False, inplace=True)
# results = results.drop(
#     ["mean_fit_time", "std_fit_time", "mean_score_time", "std_score_time", "std_test_score", "params"], axis=1)
# pd.set_option('display.max_columns', None)
# print(f"Results: {results.head(10)}")
