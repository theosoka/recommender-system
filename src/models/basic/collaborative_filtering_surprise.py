import logging
import sys

import sklearn

import pandas as pd
from surprise import Reader, Dataset
from surprise.model_selection import train_test_split
from surprise import accuracy
from src.models.basic.surprise_models_repository import SurpriseModelsRepository
from surprise.model_selection import GridSearchCV

from ..model_mixin import ModelMixin

logger = logging.getLogger("cf_surprise")
logger.setLevel(logging.INFO)

stdout_handler = logging.StreamHandler(stream=sys.stdout)
stdout_handler.addFilter(lambda rec: rec.levelno <= logging.INFO)
logger.addHandler(stdout_handler)


class CollaborativeFiltering(ModelMixin):
    def __init__(self, data: pd.DataFrame, model_name: str, test_size=0.2):
        super().__init__(model_name=model_name, dataset=data)
        self.estimator = None
        reader = Reader(line_format="user item rating", sep=" ")
        self.data = Dataset.load_from_df(self.dataset, reader)
        self.train_set, self.test_set = train_test_split(self.data, test_size=test_size)
        self.train, self.test = sklearn.model_selection.train_test_split(
            self.dataset, test_size=0.2, random_state=12
        )
        self.train_data = Dataset.load_from_df(self.train, reader)
        self.model, self.params_grid = getattr(
            SurpriseModelsRepository[model_name], "value"
        )

    def tune_and_fit(self):
        if self.params_grid:
            gs = GridSearchCV(
                self.model, self.params_grid, measures=["rmse", "mae"], cv=10, n_jobs=-1
            )
            self.estimator = gs.fit(self.train_data)
            logger.info(
                f"Grid Search for {self.model} best score is: {gs.best_score['rmse']}"
            )
            logger.info(f"Best params: {gs.best_params['rmse']}")
        else:
            self.estimator = self.model().fit(self.train_set)
            logger.info(f"RMSE: {self.calculate_accuracy()}")

    def predict(self, user_id, artist_id):
        prediction = self.estimator.predict(user_id, artist_id)
        return prediction.est

    def calculate_accuracy(self):
        predictions = self.estimator.test(self.test_set)
        return accuracy.rmse(predictions)


dataset = pd.read_csv("/data/processed/lastfm_2k/user_artists.csv")
models = [
    "SLOPE",
    "BASELINE",
]
for model in models:
    logger.info(f"Model: {model}")
    cf_model = CollaborativeFiltering(dataset, model)
    cf_model.tune_and_fit()
