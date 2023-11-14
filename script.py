# AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from tree import DecisionTree

data = pd.read_csv("data/cardio_train_medium.csv", sep=";")

data["age_years"] = np.floor(data["age"] / 365.25)
new_data = pd.get_dummies(data, columns=['cholesterol', 'gluc'])

new_data = new_data.drop(['id', 'cardio', 'age'], axis=1)
data, target = new_data.to_numpy(), np.array(data["cardio"])

X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.3, random_state=2023,
)

print(f"{X_train=}")
print(f"{X_test=}")
print(f"{y_train=}")
print(f"{y_test=}")

clf = DecisionTree(max_depth=3)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("\nAccuracy:", np.sum(y_test == y_pred) / len(y_test), "\n")
print(f"{y_test=}")
print(f"{y_pred=}")
