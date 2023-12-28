import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.svm import SVC

from sklearn.model_selection import (
    GridSearchCV,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)

sex_mapping = {'female': 0, 'male': 1}

titanik_df = pd.read_csv('titanic_train.csv')
titanik_df = titanik_df.drop(columns=['PassengerId', 'Name', 'Ticket','Cabin'])
titanik_df = titanik_df.dropna(subset=['Age'])
titanik_df['Sex']= titanik_df['Sex'].replace(sex_mapping)

unique_embarks = set(titanik_df['Embarked'])
index_by_embark = {label: idx for idx, label in enumerate(unique_embarks)}
titanik_df['Embarked'] = np.array([index_by_embark[label] for label in titanik_df['Embarked']])

# Splitting
X = np.array(titanik_df[titanik_df.columns[1:]])
y = np.array(titanik_df['Survived'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

logreg = LogisticRegression()
logreg_params = {'C': [0.1, 1, 10]}  # regularization parameter, inverse coeff
logreg_grid = GridSearchCV(logreg, logreg_params, cv=5, scoring='accuracy')
logreg_grid.fit(X_train, y_train)

results = pd.DataFrame(logreg_grid.cv_results_)
print("GridSearchCV Results for Logistic Regression:")
print(results[['param_C', 'mean_test_score', 'std_test_score', 'rank_test_score']])
print()

best_logreg_model = logreg_grid.best_estimator_
logreg_predictions = best_logreg_model.predict(X_test)
accuracy = accuracy_score(y_test, logreg_predictions)
print("Best Logistic Regression Model:", best_logreg_model)
print("Accuracy on Test Data:", accuracy)
print()

####################################################################################################
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_test[:, 2], X_test[:, 5], c=logreg_predictions, cmap='viridis', marker='o', s=50, edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Predicted Classes')

plt.subplot(1, 2, 2)
plt.scatter(X_test[:, 2], X_test[:, 5], c=y_test, cmap='viridis', marker='o', s=50, edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Actual Classes')

plt.tight_layout()
plt.show()

####################################################################################################
ridge_classifier = RidgeClassifier()
ridge_params = {'alpha': [0.1, 1, 10]}
ridge_grid = GridSearchCV(ridge_classifier, ridge_params, cv=5, scoring='accuracy')
ridge_grid.fit(X_train, y_train)

results = pd.DataFrame(ridge_grid.cv_results_)
print("GridSearchCV Results for Ridge Classification:")
print(results[['param_alpha', 'mean_test_score', 'rank_test_score']])
print()

best_ridge_classifier = ridge_grid.best_estimator_
ridge_predictions = best_ridge_classifier.predict(X_test)
accuracy = accuracy_score(y_test, ridge_predictions)

print("Best Ridge Classifier Model:", best_ridge_classifier)
print("Accuracy on Test Data:", accuracy)
print()

####################################################################################################
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_test[:, 2], X_test[:, 5], c=ridge_predictions, cmap='viridis', marker='o', s=50, edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Predicted Classes')

plt.subplot(1, 2, 2)
plt.scatter(X_test[:, 2], X_test[:, 5], c=y_test, cmap='viridis', marker='o', s=50, edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Actual Classes')

plt.tight_layout()
plt.show()

####################################################################################################
# Support Vector Classifier
svm = SVC()
svm_params = {'C': [0.1, 1, 10]}
svm_grid = GridSearchCV(svm, svm_params, cv=5, scoring='accuracy')
svm_grid.fit(X_train, y_train)

results = pd.DataFrame(svm_grid.cv_results_)
print("GridSearchCV Results for SVM:")
print(results[['param_C', 'mean_test_score', 'std_test_score', 'rank_test_score']])

best_svm_model = svm_grid.best_estimator_
svm_predictions = best_svm_model.predict(X_test)
accuracy = accuracy_score(y_test, svm_predictions)

print("Best SVM Model:", best_svm_model)
print("SVM Accuracy on Test Data:", accuracy)

####################################################################################################
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_test[:, 2], X_test[:, 5], c=svm_predictions, cmap='viridis', marker='o', s=50, edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Predicted Classes')

plt.subplot(1, 2, 2)
plt.scatter(X_test[:, 2], X_test[:, 5], c=y_test, cmap='viridis', marker='o', s=50, edgecolors='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Actual Classes')

plt.tight_layout()
plt.show()
