import pandas as pd
from catboost import CatBoostRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score, cross_validate

# Import the data
df = pd.read_csv("Concrete_Data_Yeh.csv")
X = df.drop("csMPa", axis=1)
y = df["csMPa"].copy()

# Final cross validation scoring with best parameters
print("""Best Model: CatBoostRegressor
Best Hyperparameters:  
    iterations: 10,000
    depth: 7
    learning_rate: 0.2, 
    l2_leaf_reg: 3 """)

regressor = CatBoostRegressor(iterations=10000, depth=7, learning_rate=0.2, l2_leaf_reg=3, random_state=42, verbose=0)
out = cross_validate(regressor, X, y, cv=5, scoring="neg_root_mean_squared_error", return_estimator=True)
scores = out['test_score']

# Print scores
print(f"5-Fold cross validation scores: \n\t{-scores}")
print(f"Average score: \n\t{-scores.mean()}")

# Use permutation to find the importance of each feature.
regressor.fit(X, y, verbose=0)

out = permutation_importance(regressor, X, y, scoring="neg_root_mean_squared_error")
importance = out.importances_mean

print("\n\tFeature importance: ")
for idx, est in enumerate(importance):
    print('Feature: %s, Score: %.5f' % (df.columns[idx], est))
