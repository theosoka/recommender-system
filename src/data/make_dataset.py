# -*- coding: utf-8 -*-
import sys

import click
import logging
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
import numpy as np
from enum import Enum

from utils import make_data_binary, min_max_scaler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

stdout_handler = logging.StreamHandler(stream=sys.stdout)
stdout_handler.addFilter(lambda rec: rec.levelno <= logging.INFO)
logger.addHandler(stdout_handler)


class Datasets(Enum):
    LASTFM_2K = "lastfm_2k"
    LASTFM_1K = "lastfm_1k"
    SPOTIFY = "spotify"
    SPOTIFY_JSONS = "spotify_jsons"


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
@click.argument("dataset_name", type=click.STRING)
def main(input_filepath, output_filepath, dataset_name):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    processed_dataframes = {}

    logger.info(f"Chosen dataset to process: {dataset_name}")
    match dataset_name:
        case Datasets.LASTFM_2K.value:
            processed_dataframes = make_lastfm_2k_dataset(input_filepath)
        case Datasets.LASTFM_1K.value:
            processed_dataframes = make_lastfm_1k_dataset(input_filepath)
        case Datasets.SPOTIFY.value:
            processed_dataframes = make_spotify_dataset(input_filepath)
        case Datasets.SPOTIFY_JSONS.value:
            processed_dataframes = make_spotify_jsons_dataset(input_filepath)
        case _:
            logger.info(f"{dataset_name} is not valid.")

    for name, df in processed_dataframes.items():
        filename = f"{name}.csv"
        df.to_csv(Path(output_filepath) / f"{dataset_name}/{filename}", index=False)


def make_lastfm_2k_dataset(input_filepath: str) -> dict:
    user_artists = pd.read_csv(
        Path(input_filepath) / "hetrec2011-lastfm-2k/user_artists.dat", sep="\t"
    )
    logger.info("Removing skewness in user_artists.weight")

    user_artists.weight = min_max_scaler(user_artists, "weight")
    user_friends = pd.read_csv(
        Path(input_filepath) / "hetrec2011-lastfm-2k/user_friends.dat", sep="\t"
    )

    user_taggedartists = pd.read_csv(
        Path(input_filepath) / "hetrec2011-lastfm-2k/user_taggedartists-timestamps.dat",
        sep="\t",
    )

    tags = pd.read_csv(Path(input_filepath) / "hetrec2011-lastfm-2k/tags.dat", sep="\t", encoding="latin-1")

    artists = pd.read_csv(
        Path(input_filepath) / "hetrec2011-lastfm-2k/artists.dat",
        sep="\t",
        encoding="latin-1",
    )
    logger.info("Dropping rows with null values.")
    artists = artists.dropna()

    processed_dataframes = {
        "user_artists": user_artists,
        "user_friends": user_friends,
        "user_taggedartists": user_taggedartists,
        "tags": tags,
        "artists": artists,
    }

    return processed_dataframes


def make_lastfm_1k_dataset(input_filepath: str) -> dict:
    user_profiles = pd.read_csv(Path(input_filepath) / "lastfm_1k/userid-profile.tsv")
    user_tracks = pd.read_csv(
        Path(input_filepath)
        / "lastfm_1k/userid-timestamp-artid-artname-traid-traname.tsv"
    )

    return {}


def make_spotify_dataset(input_filepath: str) -> dict:
    track_features = pd.read_csv(Path(input_filepath) / "track_features/tf_mini.csv")
    training_set = pd.read_csv(Path(input_filepath) / "training_set/log_mini.csv")

    return {}


def make_spotify_jsons_dataset(input_filepath: str) -> dict:
    return {}


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
