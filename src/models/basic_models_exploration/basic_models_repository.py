from enum import Enum

from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


class BasicModelsRepository(Enum):
    PERCEPTRON = [
        Perceptron(),
        {
            "penalty": [None, "l2", "l1", "elasticnet"],
            "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10],
            "max_iter": [100, 200, 300, 400, 500],
            "tol": [1e-3, 1e-4, 1e-5],
        },
    ]
    GNB = [
        (
            GaussianNB(),
            None,
        ),
    ]
    MNB = [
        MultinomialNB(),
        {
            "alpha": [0.1, 0.5, 1.0, 1.5, 2.0],
            "fit_prior": [True, False],
        },
    ]
    SGD_C = [
        SGDClassifier(),
        {
            "alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
            "max_iter": [900, 1000],
            "loss": ["log_loss", "hinge", "squared_error"],
            "penalty": ["l2"],
            "n_jobs": [-1],
        },
    ]
    DTC = [
        DecisionTreeClassifier(),
        {
            "criterion": ["gini", "entropy"],
            "max_depth": [None, 10, 20, 30, 40, 50],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
    ]
    RFC = [
        RandomForestClassifier(),
        {
            "n_estimators": [10, 50, 100, 200],
            "criterion": ["gini", "entropy"],
            "max_depth": [3, 5, 10, None],
            "min_samples_split": [1, 2, 4],
            "min_samples_leaf": [1, 2, 4],
        },
    ]
    GBC = [
        GradientBoostingClassifier(),
        {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 4, 5],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
    ]
