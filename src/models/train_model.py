from dataclasses import dataclass

from sklearn.model_selection import train_test_split

import pandas as pd


@dataclass
class BaseModel:
    dataset: pd.DataFrame
    X: pd.DataFrame
    y: pd.DataFrame
    X_test: pd.DataFrame
    X_train: pd.DataFrame
    y_test: pd.DataFrame
    y_train: pd.DataFrame

    def __init__(self):
        self.split_into_features_and_outcome()
        self.split_train_test()

    def split_into_features_and_outcome(self) -> None:
        self.X, self.y = self.dataset[:, :-1], self.dataset[:, -1]

    def split_train_test(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=12
        )
