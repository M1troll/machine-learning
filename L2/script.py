import pandas as pd
from pathlib import Path
from termcolor import cprint
import operator
from collections import Counter

DATA_PATH = Path("L2/titanic.csv")


def read_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read data from csv file."""
    return pd.read_csv(path)

def get_most_popular_women_name(data: pd.DataFrame) -> str:
    passanger_names: pd.Series[str] = data['Name']
    
    first_names = []
    for full_name  in passanger_names:
        if "Miss. " not in full_name and "Mrs. " not in full_name:
            continue

        if "Miss. " in full_name:
            # Example: 'Miss. Florence Briggs Thayer'
            full_name = full_name.split("Miss. ")[-1]
        elif "Mrs. " in full_name:
            # When a woman marries, her name is placed in parentheses
            # after her husband's name
            # Example: 'Cumings, Mrs. John Bradley (Florence Briggs Thayer)'
            full_name = full_name.split("(")[-1][:-1]        
        
        # First name always first
        first_names.append(full_name.split(" ")[0])

    return Counter(first_names).most_common(1)[0][0]


def main():
    data = read_data()
    
    people_count = data['PassengerId'].count()
    cprint(f"Passengers: ", "green", end="")
    print(people_count)

    men_count = data.query("Sex == 'male'")["Sex"].count()
    women_count = data.query("Sex == 'female'")["Sex"].count()
    cprint(f"On the ship: ", "green", end="")
    print(f"Men - {men_count}, Women - {women_count}")

    firs_class_passengers_count = data.query("Pclass == 1")["PassengerId"].count()
    cprint(f"First class passengers: ", "green", end="")
    print(firs_class_passengers_count)

    survived_by_all = data.query("Survived == 1")["PassengerId"].count() / people_count
    cprint(f"Probability of survival: ", "green", end="")
    print(survived_by_all)

    survived_first_class_by_all_first_class = data.query("Survived == 1 and Pclass == 1")["PassengerId"].count() / firs_class_passengers_count
    cprint(f"Probability of survival for 1 class people: ", "green", end="")
    print(survived_first_class_by_all_first_class)

    survived_men_by_all = data.query("Survived == 1 and Sex == 'male'")["PassengerId"].count() / people_count
    cprint(f"Probability of survival for men: ", "green", end="")
    print(survived_men_by_all)

    survived_women_by_all = data.query("Survived == 1 and Sex == 'female'")["PassengerId"].count() / people_count
    cprint(f"Probability of survival for women: ", "green", end="")
    print(survived_women_by_all)

    most_popular_women_name = get_most_popular_women_name(data)
    cprint(f"Most popular women name is - ", "green", end="")
    cprint(most_popular_women_name, "red")


if __name__ == "__main__":
    main()
