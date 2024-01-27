import numpy as np
import pytest
import pandas as pd
from src.data.utils import make_data_binary, min_max_scaler, drop_outliners


@pytest.fixture
def sample_dataframe():
    data = {"A": [1, 5, 10, 15, 20], "B": [0.5, 1.5, 5.5, 10.5, 15.5]}
    return pd.DataFrame(data)


@pytest.mark.parametrize(
    "threshold, column, expected_result",
    [
        (
            10,
            "A",
            pd.DataFrame({"A": [0, 0, 1, 1, 1], "B": [0.5, 1.5, 5.5, 10.5, 15.5]}),
        ),
        (5, "B", pd.DataFrame({"A": [1, 5, 10, 15, 20], "B": [0, 0, 1, 1, 1]})),
        (
            25,
            "A",
            pd.DataFrame({"A": [0, 0, 0, 0, 0], "B": [0.5, 1.5, 5.5, 10.5, 15.5]}),
        ),
    ],
)
def test_make_data_binary(sample_dataframe, threshold, column, expected_result):
    result = make_data_binary(df=sample_dataframe, threshold=threshold, column=column)
    expected_result = expected_result.astype(result.dtypes)
    pd.testing.assert_frame_equal(result, expected_result)


@pytest.mark.parametrize(
    "column, expected_result",
    [
        (
            "A",
            pd.DataFrame(
                {
                    "A": [1.0, 1.84210526, 2.89473684, 3.94736842, 5.0],
                    "B": [0.5, 1.5, 5.5, 10.5, 15.5],
                }
            ),
        ),
        (
            "B",
            pd.DataFrame(
                {
                    "A": [1, 5, 10, 15, 20],
                    "B": [1.0, 1.26666667, 2.33333333, 3.66666667, 5.0],
                }
            ),
        ),
    ],
)
def test_min_max_scaler(sample_dataframe, column, expected_result):
    result = min_max_scaler(sample_dataframe, column)

    if isinstance(result, np.ndarray):
        result = pd.DataFrame(result, columns=[column])

    expected_result = expected_result[[column]]
    pd.testing.assert_frame_equal(result, expected_result)
