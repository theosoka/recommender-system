# -*- coding: utf-8 -*-
import sys
from copy import deepcopy

import click
import logging
from pathlib import Path
import pandas as pd
from enum import Enum

from utils import make_data_binary, min_max_scaler, drop_outliners

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
def main(dataset_name, input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    processed_dataframes = {}

    logger.info(f"Chosen dataset to process: {dataset_name}")
    match dataset_name:
        case Datasets.LASTFM_2K.value:
            processed_dataframes = make_lastfm_2k_dataset(input_filepath)
        case _:
            logger.info(f"{dataset_name} is not valid.")

    for name, df in processed_dataframes.items():
        filename = f"{name}.csv"
        df.to_csv(Path(output_filepath) / f"{dataset_name}/{filename}", index=False)


def make_lastfm_2k_dataset(input_filepath: str) -> dict:
    user_artists = pd.read_csv(
        Path(input_filepath) / "hetrec2011-lastfm-2k/user_artists.dat", sep="\t"
    )
    user_artists_prepared, user_artists_binary = make_user_artist(user_artists)

    user_friends = pd.read_csv(
        Path(input_filepath) / "hetrec2011-lastfm-2k/user_friends.dat", sep="\t"
    )
    user_tagged_artists = pd.read_csv(
        Path(input_filepath) / "hetrec2011-lastfm-2k/user_taggedartists-timestamps.dat",
        sep="\t",
    )
    user_tagged_artists.drop_duplicates()
    artist_most_popular_tag = (
        user_tagged_artists.groupby("artistID")["tagID"].agg(pd.Series.mode).to_frame()
    )

    tags = pd.read_csv(
        Path(input_filepath) / "hetrec2011-lastfm-2k/tags.dat",
        sep="\t",
        encoding="latin-1",
    )

    artists = pd.read_csv(
        Path(input_filepath) / "hetrec2011-lastfm-2k/artists.dat",
        sep="\t",
        encoding="latin-1",
    )
    artists_prepared = make_artists(artists)

    processed_dataframes = {
        "user_artists_prepared": user_artists_prepared,
        "user_artists_binary": user_artists_binary,
        "user_friends": user_friends,
        "user_tagged_artists": artist_most_popular_tag,
        "tags": tags,
        "artists": artists_prepared,
    }

    return processed_dataframes


def make_user_artist(user_artists: pd.DataFrame):
    logger.info("Dropping outliners from user_artists")
    user_artists_no_outliners = drop_outliners(df=user_artists, column="weight")
    logger.info("Normalising user_artists.weight")
    user_artists_binary = make_data_binary(
        df=user_artists_no_outliners, threshold=20, column="weight"
    )
    user_artists_no_outliners.weight = min_max_scaler(
        user_artists_no_outliners, "weight"
    )
    return user_artists_no_outliners, user_artists_binary


def make_artists(artists: pd.DataFrame):
    logger.info("Dropping unnecessary columns from artists df.")
    artists_only_features = artists.drop(columns=["pictureURL"])
    logger.info("Removing feats from artists df.")
    artists_no_feats = artists_only_features[
        artists_only_features["name"].str.contains(" ft. ") == False
    ]
    return artists_no_feats


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
