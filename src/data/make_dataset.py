# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
import numpy as np


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    logger.info('reading raw datasets')

    user_artists = pd.read_csv(Path(input_filepath) / "user_artists.dat", sep="\t")
    logger.info('removing skewness in user_artists.weight')
    qt = QuantileTransformer(n_quantiles=500, output_distribution='normal')
    user_artists.weight = pd.Series((qt.fit_transform(np.array(user_artists.weight).reshape(-1, 1))).flatten())

    user_friends = pd.read_csv(Path(input_filepath) / "user_friends.dat", sep="\t")

    user_taggedartists = pd.read_csv(Path(input_filepath) / "user_taggedartists-timestamps.dat", sep="\t")

    tags = pd.read_csv(Path(input_filepath) / "tags.dat", sep="\t", encoding='latin-1')

    artists = pd.read_csv(Path(input_filepath) / "artists.dat", sep="\t", encoding='latin-1')
    logger.info('dropping rows with null values')
    artists = artists.dropna()

    processed_dataframes = {
        "user_artists": user_artists,
        "user_friends": user_friends,
        "user_taggedartists": user_taggedartists,
        "tags": tags,
        "artists": artists,
    }

    for name, df in processed_dataframes.items():
        filename = f"{name}.csv"
        df.to_csv(Path(output_filepath) / filename, index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
