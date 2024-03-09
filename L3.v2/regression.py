import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor

import seaborn; seaborn.set()

boston_housing = df = pd.read_csv('boston_housing.csv')
X = np.array(boston_housing[boston_housing.columns[:-1]])

# medv - median value of owner-occupied homes in $1000s.
y = np.array(boston_housing[boston_housing.columns[-1]])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
knn_reg = KNeighborsRegressor()

n = 10
param_metric = {'n_neighbors': np.arange(1, n+1), 'metric': ['chebyshev','manhattan','euclidean']}

grid_search = GridSearchCV(knn_reg, param_metric, cv=5)
grid_search.fit(X_train, y_train)

results = pd.DataFrame(grid_search.cv_results_)[['param_n_neighbors', 'param_metric', 'mean_test_score']]
results_pivot = results.pivot_table(index='param_metric', columns='param_n_neighbors', values='mean_test_score')

best_params = grid_search.best_params_

best_knn_reg = KNeighborsRegressor(n_neighbors=best_params['n_neighbors'], metric=best_params['metric'])
best_knn_reg.fit(X_train, y_train)

y_pred = best_knn_reg.predict(X_test)

print("Results:")
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(results_pivot)
print()

max_values = results_pivot.max(axis=0)
n_neighbors = results_pivot.idxmax(axis=1)
print(max_values)
result_max = pd.concat([max_values, n_neighbors], axis=1, keys=['0', '1'])
result_max = result_max.reset_index()

print("Score:")
print(result_max)
print()

print("Best params:", best_params)
print("Accuracy:", grid_search.best_score_)
print()

mse = mean_squared_error(y_test, y_pred)
print("Root mean square error: ", mse)
print()


####################################################################################################
plt.figure(figsize=(12, 5))

# Prediction
plt.scatter(y_test, y_pred, edgecolors=(0, 0, 0))
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, color="red")
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')

plt.tight_layout()
plt.show()

plt.plot(results_pivot.columns, results_pivot.iloc[0], 's', color = "red", label=results_pivot.index[0])
plt.plot(results_pivot.columns, results_pivot.iloc[1], 'x', color = "black", label=results_pivot.index[1])
plt.plot(results_pivot.columns, results_pivot.iloc[2], 'd', color = "blue", label=results_pivot.index[2])

plt.legend()
plt.xticks(results_pivot.columns)
plt.ylabel('Mean cross validation resutls')
plt.xlabel('n_neighbors')

plt.show()
