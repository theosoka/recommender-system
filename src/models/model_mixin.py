import logging
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score

import pandas as pd

from basic_models_repository import BasicModelsRepository

logger = logging.getLogger("model_mixin")
logger.setLevel(logging.INFO)

stdout_handler = logging.StreamHandler(stream=sys.stdout)
stdout_handler.addFilter(lambda rec: rec.levelno <= logging.INFO)
logger.addHandler(stdout_handler)


@dataclass
class ModelMixin:
    model_name: str
    dataset: pd.DataFrame
    X: Optional[pd.DataFrame] = None
    y: Optional[pd.DataFrame] = None
    X_test: Optional[pd.DataFrame] = None
    X_train: Optional[pd.DataFrame] = None
    y_test: Optional[pd.DataFrame] = None
    y_train: Optional[pd.DataFrame] = None
    models_dump_path: Path = Path() / "../../models"

    def __post_init__(self):
        self.split_into_features_and_target()
        self.split_train_test()
        #  todo: pass with init statements

    def split_into_features_and_target(self) -> None:
        logger.info(
            f"Splitting dataset into X and y. Dataset shape: {self.dataset.shape}"
        )
        self.X, self.y = self.dataset.iloc[:, :-1], self.dataset.iloc[:, -1]

    def split_train_test(self) -> None:
        logger.info("Splitting the dataset into train and test")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=12
        )

    def _tune_and_fit_model(self, model: Any, param_grid: dict) -> Any:
        if param_grid:
            logger.info("Tuning model with Grid Search.")
            classifier = GridSearchCV(
                model,
                param_grid,
                n_jobs=1,
            )
            logger.info("Fitting the best estimator.")
            classifier.fit(self.X, self.y)
            return classifier.best_estimator_
        else:
            model.fit(self.X, self.y)
            return model

    def dump_model_into_file(self, model, name) -> None:
        file_path = self.models_dump_path / f"{name}.pkl"
        logger.info(f"Dumping model into {file_path}")
        pickle.dump(model, open(file_path, "wb"))

    def predict_and_estimate(self, model) -> None:
        y_pred = model.predict(self.X_test)
        logger.info(
            f"Model name: {self.model_name}\n"
            f"Accuracy: {accuracy_score(self.y_test, y_pred)}\n"
            f"F1-Score: {f1_score(self.y_test, y_pred)}\n"
        )

    def add_separator(self) -> str:
        return "\n====================\n\n"

    def run(self):
        logger.info(f"Fitting {self.model_name}")
        self.split_into_features_and_target()
        self.split_train_test()
        model, param_grid = getattr(BasicModelsRepository[self.model_name], "value")
        estimator = self._tune_and_fit_model(model, param_grid)
        self.predict_and_estimate(estimator)
        self.dump_model_into_file(estimator, self.model_name)
        logger.info(
            f"{self.model_name} was fit and saved into a file.{self.add_separator()}"
        )
