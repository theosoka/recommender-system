import pandas as pd
from model_mixin import ModelMixin


def train_basic_models(dataset: pd.DataFrame, models_aliases: list):
    for model in models_aliases:
        model = ModelMixin(model_name=model, dataset=dataset)
        model.run()


def main():
    models = []
    dataset = pd.read_csv("/data/processed/lastfm_2k/user_artists.csv")
    train_basic_models(dataset, models)


if __name__ == "__main__":
    main()
