import numpy as np
import pandas as pd

# Загрузка данных
data = pd.read_csv("data/cardio_train_small.csv", sep=";")

# Определение целевой переменной
target = "age"

# Разбиение данных на обучающую и тестовую выборки
train_data = data.sample(frac=0.8, random_state=42)
test_data = data.drop(train_data.index)


# Определение функции для разбиения данных
def split_data(data, feature, value):
    left = data[data[feature] < value]
    right = data[data[feature] >= value]
    return left, right

# Определение функции для вычисления энтропии
def entropy(data):
    target_col = data[target]
    _, counts = np.unique(target_col, return_counts=True)
    probs = counts / counts.sum()
    return -(probs * np.log2(probs)).sum()

# Определение функции для выбора лучшего признака
def select_feature(data, features):
    best_entropy = np.inf
    best_feature = None
    best_value = None
    for feature in features:
        for value in data[feature].unique():
            left, right = split_data(data, feature, value)
            if left.empty or right.empty:
                continue
            all_entropy = entropy(data)
            left_entropy = entropy(left)
            right_entropy = entropy(right)
            weighted_entropy = (left.shape[0] / data.shape[0]) * left_entropy + (right.shape[0] / data.shape[0]) * right_entropy
            if weighted_entropy < best_entropy:
                best_entropy = weighted_entropy
                best_feature = feature
                best_value = value
    return best_feature, best_value

# Определение функции для создания дерева решений
def create_tree(data, features, max_depth):
    if max_depth == 0 or data[target].nunique() == 1:
        target_col = data[target]
        return target_col.iat[0]
    best_feature, best_value = select_feature(data, features)
    left, right = split_data(data, best_feature, best_value)
    subtree = {best_feature: {}}
    subtree[best_feature]["< {}".format(best_value)] = create_tree(left, features, max_depth - 1)
    subtree[best_feature][">= {}".format(best_value)] = create_tree(right, features, max_depth - 1)
    return subtree

# Определение функции для предсказания
def predict(tree, observation):
    feature, value = next(iter(tree.items()))
    if observation[feature] < value["< {}".format(value)]:
        return predict(value["< {}".format(value)], observation)
    else:
        return predict(value[">= {}".format(value)], observation)


# Обучение модели и оценка ее качества
features = list(train_data.columns)
print(features)
features.remove(target)
tree = create_tree(train_data, features, max_depth=3)

predictions = []
for index, row in test_data.iterrows():
    predictions.append(predict(tree, row))
    print(f"Row #{index}")

accuracy = (predictions == test_data[target]).mean()
print("Accuracy:", accuracy)
