import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def make_data_binary(df: pd.DataFrame, threshold: int, column: str) -> pd.DataFrame:
    df.loc[df[column] >= threshold, column] = 1
    df.loc[df[column] < threshold, column] = 0
    return df


def min_max_scaler(df: pd.DataFrame, column: str) -> pd.DataFrame:
    scaler = MinMaxScaler(feature_range=(1, 5))
    column_scaled = scaler.fit_transform(df[[column]])
    return column_scaled
