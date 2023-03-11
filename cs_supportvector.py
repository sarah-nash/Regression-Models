import pandas as pd
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

df = pd.read_csv("Concrete_Data_Yeh.csv")

X = df.drop("csMPa", axis=1)
y = df["csMPa"].copy()

# Final cross validation, printing out scores
print("""Support Vector Regression -- Best Hyperparameters: 
    kernel: 'rbf', 
    epsilon: 0.0, 
    degree: 12, 
    C: 370""")  # Expected Best Score: 9.396557582195422

regressor = SVR(kernel='rbf', degree=12, epsilon=0.0, C=370)
scores = -cross_val_score(regressor, X, y, cv=5, scoring="neg_root_mean_squared_error")
print(f"5-fold cross validation scores: \n\t{scores}")
print(f"Average score: \n\t{scores.mean()}")



# Initial random parameter search to determine which ones may be significant
# regressor = SVR()
# params = {"kernel": ["linear", "poly", "rbf"],
#           "degree": randint(low=0, high=20),
#           "C": [0.1, 1, 10, 50, 100, 1000],
#           "epsilon": [0, 0.01, 0.05, 0.1, 0.5, 1]
#           }
# rs = RandomizedSearchCV(regressor, params, cv=5, scoring="neg_root_mean_squared_error", random_state=42, n_jobs=-1, n_iter=20)
# rs.fit(X, y)
# print(f"Best Params: {rs.best_params_}")
# print(f"Best Score: {-rs.best_score_}")
#
# # Code for displaying top 10 random search results
# results = pd.DataFrame(rs.cv_results_)
# results.sort_values(by="mean_test_score", ascending=False, inplace=True)
# results = results.drop(
#     ["mean_fit_time", "std_fit_time", "mean_score_time", "std_score_time", "std_test_score", "params"], axis=1)
# pd.set_option('display.max_columns', None)
# print(f"Results: {results.head(10)}")

# Secondary random search
# regressor = SVR()
# params = {"kernel": ["rbf"],
#           "degree": randint(low=5, high=15),
#           "C": [10 * i for i in range(10, 100)],
#           "epsilon": [0.01 * i for i in range(0, 50)]  # small epsilon worked better
#           }
#
# rs = RandomizedSearchCV(regressor, params, cv=5, scoring="neg_root_mean_squared_error", random_state=42, n_jobs=-1, n_iter=50)
# rs.fit(X, y)
# print(f"Best Params: {rs.best_params_}")
# print(f"Best Score: {-rs.best_score_}")
#
# # Code for displaying top 10 random search results
# results = pd.DataFrame(rs.cv_results_)
# results.sort_values(by="mean_test_score", ascending=False, inplace=True)
# results = results.drop(
#     ["mean_fit_time", "std_fit_time", "mean_score_time", "std_score_time", "std_test_score", "params"], axis=1)
# pd.set_option('display.max_columns', None)
# print(f"Results: {results.head(10)}")

# Final grid search
# regressor = SVR()
# params = {"kernel": ["rbf"],
#           "degree": [10, 11, 12, 13, 14],
#           "C": [10 * i for i in range(30, 40)],
#           "epsilon": [0.01 * i for i in range(0, 20)]  # small epsilon worked better
#           }
#
# rs = RandomizedSearchCV(regressor, params, cv=5, scoring="neg_root_mean_squared_error", random_state=42, n_jobs=-1,
#                         n_iter=50)
# rs.fit(X, y)
# print(f"Best Params: {rs.best_params_}")
# print(f"Best Score: {-rs.best_score_}")

