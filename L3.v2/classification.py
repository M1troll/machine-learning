import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier

import seaborn; seaborn.set()


iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
knn = KNeighborsClassifier()

n = 10
param_grid = {'n_neighbors': np.arange(1, n+1), 'metric': ['chebyshev','manhattan','euclidean']}
grid_search = GridSearchCV(knn, param_grid, cv=n)
grid_search.fit(X_train, y_train)

results = pd.DataFrame(grid_search.cv_results_)[['param_n_neighbors', 'param_metric', 'mean_test_score']]
results_pivot = results.pivot_table(index='param_metric', columns='param_n_neighbors', values='mean_test_score')

best_params = grid_search.best_params_

best_knn = KNeighborsClassifier(n_neighbors=best_params['n_neighbors'], metric=best_params['metric'])
best_knn.fit(X_train, y_train)

y_pred = best_knn.predict(X_test)
print("Results:")
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(results_pivot)
print()

max_values = results_pivot.max(axis=1)
n_neighbors = results_pivot.idxmax(axis=1)
result_max = pd.concat([max_values, n_neighbors], axis=1, keys=['mean_test_score', 'n_neighbors'])
result_max = result_max.reset_index()

print("Score:")
print(result_max)
print()

print("Best params:", best_params)
print("Best score:", grid_search.best_score_)
print()


####################################################################################################
plt.figure(figsize=(12, 5))

# Predication
plt.subplot(1, 2, 1)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, marker='o', edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Predicted Classes')

# Real state
plt.subplot(1, 2, 2)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='o', edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Actual Classes')

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
