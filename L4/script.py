import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import (
    GridSearchCV,
    cross_val_score,
    train_test_split,
)

boston_housing = pd.read_csv('boston_housing.csv')
X = np.array(boston_housing[boston_housing.columns[:-1]])

# medv - median value of owner-occupied homes in $1000s.
y = np.array(boston_housing[boston_housing.columns[-1]])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Кратко прочитай про разницу регрессий линейной, риджа и лассо 
linear_reg = LinearRegression()
# что такое CV и почему именно 5?
cv_scores = cross_val_score(linear_reg, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
# почему берем отричательное значение (это как-то относится к приставке neg_ выше)
cv_scores = -cv_scores

print("Cross-Validation Scores:", cv_scores)
print("Mean MSE:", np.mean(cv_scores))
print()

linear_reg.fit(X_train, y_train)
linear_predictions = linear_reg.predict(X_test)

linear_mse = mean_squared_error(y_test, linear_predictions)
print("Linear Regression MSE on Test Data:", linear_mse)
print()

ridge = Ridge()
ridge_params = {'alpha': [0.1, 1, 10]} # search params
ridge_grid = GridSearchCV(ridge, ridge_params, cv=5, scoring='neg_mean_squared_error')
ridge_grid.fit(X_train, y_train)

results = pd.DataFrame(ridge_grid.cv_results_)
print("GridSearchCV Results for Ridge Regression:")
print(results[['param_alpha', 'mean_test_score', 'std_test_score', 'rank_test_score']])
print()

best_ridge_model = ridge_grid.best_estimator_
ridge_predictions = best_ridge_model.predict(X_test)
ridge_mse = mean_squared_error(y_test, ridge_predictions)

print("Best Ridge Model:", best_ridge_model)
print("Ridge Regression MSE on Test Data:", ridge_mse)
print()

lasso = Lasso()
lasso_params = {'alpha': [0.1, 1, 10]}
lasso_grid = GridSearchCV(lasso, lasso_params, cv=5, scoring='neg_mean_squared_error')
lasso_grid.fit(X_train, y_train)

results = pd.DataFrame(lasso_grid.cv_results_)
print("GridSearchCV Results for Lasso Regression:")
print(results[['param_alpha', 'mean_test_score', 'std_test_score', 'rank_test_score']])

best_lasso_model = lasso_grid.best_estimator_
lasso_predictions = best_lasso_model.predict(X_test)
lasso_mse = mean_squared_error(y_test, lasso_predictions)

print("Best Lasso Model:", best_lasso_model)
print("Lasso Regression MSE on Test Data:", lasso_mse)
print()


############################## Comparing ######################################
results_df = pd.DataFrame({
    'Actual': y_test,
    'Linear Regression': linear_predictions,
    'Ridge Regression': ridge_predictions,
    'Lasso Regression': lasso_predictions
})
results_df['RowNumber'] = results_df.index

plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
plt.plot(results_df['RowNumber'], results_df['Linear Regression'], color='r', label='Linear Regression')
plt.plot(results_df['RowNumber'],results_df['Actual'], color='g', label='Actual')
plt.title('Linear Regression')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(results_df['RowNumber'], results_df['Ridge Regression'], color='r', label='Ridge Regression')
plt.plot(results_df['RowNumber'],results_df['Actual'], color='g', label='Actual')
plt.title('Ridge Regression')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(results_df['RowNumber'], results_df['Lasso Regression'], color='r', label='Lasso Regression')
plt.plot(results_df['RowNumber'],results_df['Actual'], color='g', label='Actual')
plt.title('Lasso Regression')
plt.legend()

plt.tight_layout()
plt.show()
