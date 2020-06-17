import statistics
from typing import Tuple

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import KFold

import src.testing.SvmParameters as Params
from src.classifiers import PCA


def __split_features(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X = features[:, :-1]
    y = features[:, -1]
    return X, y


# noinspection PyPep8Naming
def test(features: np.ndarray, clf, random_state: int = Params.RANDOM_STATE, activities: np.ndarray = None) -> float:
    X, y = __split_features(features)
    scores = []

    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train, X_test = Params.PREPROCESSOR.preprocess(X_train, X_test)
        X_train, X_test = PCA.decompose(X_train, X_test)

        clf.fit(X_train, y_train)
        predict = clf.predict(X_test)

        if activities is not None:
            print("Report:\n", classification_report(y_test, predict, target_names=activities))
            print("Confusion matrix:\n", confusion_matrix(y_test, predict), end="\n\n")

        acc_score = accuracy_score(y_test, predict)
        scores.append(acc_score)

    return statistics.mean(scores) * 100
