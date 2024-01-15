from enum import Enum
from surprise import KNNBasic, SVD, SVDpp, NMF, SlopeOne, BaselineOnly


class SurpriseModelsRepository(Enum):
    KNN = (
        KNNBasic,
        {
            "k": [5, 10, 20],
            "sim_options": {
                "name": ["cosine"],
            },
        },
    )

    SVD = (
        SVD,
        {
            "n_epochs": [5, 10, 20],
            "n_factors": [20, 50, 100],
            "lr_all": [0.002, 0.005],
            "reg_all": [0.4, 0.6],
        },
    )

    SVDPP = (
        SVDpp,
        {
            "n_epochs": [5, 10, 20],
            "n_factors": [20, 50, 100],
            "lr_all": [0.002, 0.005],
            "reg_all": [0.4, 0.6],
        },
    )

    NMF = (
        NMF,
        {
            "n_epochs": [5, 10, 20],
            "lr_bu": [0.002, 0.005],
            "lr_bi": [0.002, 0.005],
            "reg_bu": [0.4, 0.6],
            "reg_bi": [0.4, 0.6],
        },
    )

    SLOPE = (SlopeOne, None)

    BASELINE = (BaselineOnly, None)
