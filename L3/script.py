import pandas as pd
import numpy as np
import mglearn
import matplotlib.pyplot as plt
import seaborn
from termcolor import cprint
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from typing import Any

seaborn.set()

NEIGHBORS = 6
BLOCKS_COUNT = 5
GRID_PARAMS = [
    {
        "metric": [
            "chebyshev",
            "manhattan",
            "euclidean",
        ],
        "n_neighbors": range(1, NEIGHBORS + 1)
    },
]



def my_train_test_split(
    x_class: pd.DataFrame,
    y_class: pd.DataFrame,
    test_size: float = 0.3,
    random_state: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Override to add type hints."""
    return train_test_split(
        x_class,
        y_class,
        test_size=test_size,
        random_state=random_state,
    )    # type: ignore

def print_with_title(
    title: str,
    text: Any,
    title_color: str = "green",
    text_color: str = "white",
):
    """Make colored print."""
    cprint(f"{title}: ", title_color, end="")
    cprint(str(text), text_color)


def main():
    dataset = pd.read_csv("L3/iris.csv", sep=",")
    x_class, y_class = np.hsplit(dataset, [4])
    x_train, x_test, y_train, y_test = my_train_test_split(x_class, y_class)   # type: ignore

    n_train, _ = x_train.shape
    n_test, _ = x_test.shape
    print_with_title("Train", n_train)
    print_with_title("Test", n_test)

    knn = KNeighborsClassifier()
    grid = GridSearchCV(knn, GRID_PARAMS, cv=BLOCKS_COUNT)

    # Education
    grid.fit(x_train, y_train.iloc[:,0])
    # Estimate
    grid.score(x_test, y_test.iloc[:,0])
    
    print_with_title(f"Best params values", grid.best_params_)
    print_with_title(f"Best cross-validation", grid.best_score_)
    print_with_title(f"Best models", grid.best_estimator_)

    
    result =  pd.DataFrame()
    result["n_neighbors"] = grid.cv_results_["param_n_neighbors"]
    result["metric"] = grid.cv_results_["param_metric"]
    result["mean_test_score"] = grid.cv_results_["mean_test_score"]

    # First metric
    first_metric_neighbors = result["n_neighbors"][0:NEIGHBORS]
    first_metric_score = result["mean_test_score"][0:NEIGHBORS]
    plt.plot(
        first_metric_neighbors,
        first_metric_score,
        "d",
        color="red",
        label=result["metric"][0],
    )

    # Second metric
    second_metric_neighbors = result["n_neighbors"][NEIGHBORS:NEIGHBORS * 2]
    second_metric_score = result["mean_test_score"][NEIGHBORS:NEIGHBORS * 2]
    plt.plot(
        second_metric_neighbors,
        second_metric_score,
        "d",
        color="cyan",
        label=result["metric"][NEIGHBORS],
    )
    
    # Third metric
    second_metric_neighbors = result["n_neighbors"][NEIGHBORS * 2:NEIGHBORS * 3]
    second_metric_score = result["mean_test_score"][NEIGHBORS * 2:NEIGHBORS * 3]
    plt.plot(
        second_metric_neighbors,
        second_metric_score,
        "d",
        color="green",
        label=result["metric"][NEIGHBORS * 2],
    )

    plt.ylabel("Mean value based on cross-validation results")
    plt.xlabel("n_neighbors")
    # plt.show()

    best_ch = 0
    best_man = 0
    best_mink = 0
    j=0

    # Add counter
    # range for 'j' is taken from result metric row

    #j-номер строки
    for j in range(6):

        if result['mean_test_score'][j] > best_ch:
            best_ch = result['mean_test_score'][j]
            k1=result['metric'][j]
            k2=result['n_neighbors'][j]

    for j in range(6, 12):
        if result['mean_test_score'][j] > best_man:
            best_man = result['mean_test_score'][j]
            k3=result['metric'][j]
            k4=result['n_neighbors'][j]
            
    for j in range(12, 18):
        if result['mean_test_score'][j] > best_mink:
            best_mink = result['mean_test_score'][j]
            k5=result['metric'][j]
            k6=result['n_neighbors'][j]

    print_with_title(title="Best mean score of 'chebyshev'", text=best_ch)
    print_with_title(title="Best mean score of 'manhattan'", text=best_man)
    print_with_title(title="Best mean score of 'euclidean'", text=best_mink)

    knn1 = KNeighborsClassifier(metric=k1, n_neighbors=k2)    
    knn1.fit(x_train, y_train)
    knn1.score(x_test, y_test)
    print(knn1.score(x_test, y_test))

    knn2 = KNeighborsClassifier(metric=k3, n_neighbors=k4)    
    knn2.fit(x_train, y_train)
    knn2.score(x_test, y_test)
    print(knn2.score(x_test, y_test))

    knn3 = KNeighborsClassifier(metric=k5, n_neighbors=k6)    
    knn3.fit(x_train, y_train)
    knn3.score(x_test, y_test)
    print(knn3.score(x_test, y_test))

    plt.plot(k2, best_ch, 's', color = "orange", label=k1)
    plt.plot(k4, best_man, 's', color = "red", label=k3)
    plt.plot(k6, best_mink, 's', color = "black", label=k5)

    plt.ylabel('Точность')
    plt.xlabel('n_neighbors')
    plt.show()


if __name__ == "__main__":
    main()
