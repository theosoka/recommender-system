from copy import deepcopy

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def make_data_binary(df: pd.DataFrame, threshold: int, column: str) -> pd.DataFrame:
    new_df = deepcopy(df)
    new_df.loc[df[column] < threshold, column] = 0
    new_df.loc[df[column] >= threshold, column] = 1
    return new_df


def min_max_scaler(df: pd.DataFrame, column: str) -> pd.DataFrame:
    scaler = MinMaxScaler(feature_range=(1, 5))
    column_scaled = scaler.fit_transform(df[[column]])
    return column_scaled


def drop_outliners(df: pd.DataFrame, column: str, factor=3):
    upper_lim = df[column].mean() + df[column].std() * factor
    lower_lim = df[column].mean() - df[column].std() * factor
    processed_df = df[(df[column] < upper_lim) & (df[column] > lower_lim)]
    return processed_df
