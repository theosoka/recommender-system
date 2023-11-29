from enum import Enum

from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


class ModelNames(Enum):
    PERCEPTRON = "pn"
    GAUSSIAN_NAIVE_BAYES = "gnb"
    MULTINOMIAL_NB = "mnb"
    SVC = "svc"
    SGD_CLASSIFIER = "sgd_c"
    DECISION_TREE_CLASSIFIER = "dtc"
    RANDOM_FOREST_CLASSIFIER = "rfc"
    GRADIENT_BOOSTING_CLASSIFIER = "gbc"


class BasicModelsRepository(Enum):
    PERCEPTRON = (
        (
            Perceptron(),
            {
                "penalty": [None, "l2", "l1", "elasticnet"],
                "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10],
                "max_iter": [100, 200, 300, 400, 500],
                "tol": [1e-3, 1e-4, 1e-5],
            },
        ),
    )
    GNB = (
        (
            GaussianNB(),
            None,
        ),
    )
    MNB = (
        MultinomialNB(),
        {
            "alpha": [0.1, 0.5, 1.0, 1.5, 2.0],
            "fit_prior": [True, False],
        },
    )
    SVC = (
        (
            SVC(),
            {
                "C": [0.1, 1, 10, 100],
                "kernel": ["linear", "rbf", "poly", "sigmoid"],
                "gamma": ["scale", "auto"],
                "degree": [2, 3, 4, 5],
                "coef0": [0.0, 0.1, 0.5, 1.0],
            },
        ),
    )
    SGD_C = (
        (
            SGDClassifier(),
            {
                "loss": ["hinge", "log", "modified_huber", "squared_hinge"],
                "penalty": ["l1", "l2", "elasticnet"],
                "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10],
                "learning_rate": ["constant", "optimal", "invscaling", "adaptive"],
                "max_iter": [900, 1000, 2000],
                "tol": [1e-3, 1e-4, 1e-5],
            },
        ),
    )
    DTC = (
        (
            DecisionTreeClassifier(),
            {
                "criterion": ["gini", "entropy"],
                "max_depth": [None, 10, 20, 30, 40, 50],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
        ),
    )
    RFC = (
        (
            RandomForestClassifier(),
            {
                "n_estimators": [50, 100, 200],
                "criterion": ["gini", "entropy"],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
        ),
    )
    GBC = (
        GradientBoostingClassifier(),
        {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 4, 5],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
    )
