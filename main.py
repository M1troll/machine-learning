from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

TRAIN_DATA_FILE = Path("train.csv")
TEST_DATA_FILE = Path("test.csv")
RESULT_DATA_FILE = Path("decision_tree.csv")

RESEARCH_COLUMNS = ["Pclass", "Sex", "Age"]
FIND_COLUMN = "Survived"
RESULT_COLUMNS = {"PassengerId": [], "Survived": []}


def change_column_values(
    data: pd.DataFrame,
    column_name: str,
    values: dict[str, Any],
) -> pd.DataFrame:
    """Change values in DataFrame column by name.
    
    Get values argument. It's dict with new values for column in next format:
        key - previous value of cell
        value - new value of cell

    """
    new_data = data.copy(deep=True)
    for key, value in values.items():
        new_data.loc[new_data[column_name] == key, column_name] = value
    return new_data


def make_predict(
        input_data: list[list[Any]],
        train_data: pd.DataFrame,
    ) -> np.array:  # type: ignore
    """Performs prediction by the decision tree method."""
    main_columns = train_data[RESEARCH_COLUMNS].fillna(0)
    survived_column = train_data[FIND_COLUMN].fillna(0)

    tree = DecisionTreeClassifier()
    tree.fit(main_columns, survived_column)
    result = tree.predict(input_data)

    return result.item()

def main():
    """Performs passenger mortality prediction based on a training sample."""
    train_data = change_column_values(
        data = pd.read_csv(TRAIN_DATA_FILE),
        column_name="Sex",
        values={"male": 1, "female": 0},
    )
    test_data = change_column_values(
        data = pd.read_csv(TEST_DATA_FILE),
        column_name="Sex",
        values={"male": 1, "female": 0},
    )
    result_data = pd.DataFrame(RESULT_COLUMNS)

    for _, row in test_data.iterrows():
        input_data = row[RESEARCH_COLUMNS].fillna(0).to_list()
        result_data.loc[len(result_data.index)] = [     # type: ignore
            row["PassengerId"],
            make_predict([input_data], train_data),
        ]
    
    print(result_data)
    result_data.to_csv(RESULT_DATA_FILE, index = False)
    print("Results were saved at file - ", RESULT_DATA_FILE)


if __name__ == "__main__":
    main()
    # https://www.kaggle.com/competitions/titanic/submissions#
