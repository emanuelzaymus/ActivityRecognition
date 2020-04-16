import statistics
from enum import Enum
from typing import Tuple

import numpy as np

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, Normalizer, RobustScaler, MinMaxScaler, MaxAbsScaler


class Preprocessing(Enum):
    NOTHING = None

    STANDARD_SCALER = StandardScaler()
    ROBUST_SCALER = RobustScaler()
    MIN_MAX_SCALER = MinMaxScaler()
    MAX_ABS_SCALER = MaxAbsScaler()

    NORMALIZER_L1 = Normalizer('l1')
    NORMALIZER_L2 = Normalizer('l2')
    NORMALIZER_MAX = Normalizer('max')

    def preprocess(self, X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # print(self.name)

        if self is not Preprocessing.NOTHING:
            train_ = self.value.fit_transform(X_train)
            test_ = self.value.transform(X_test)
            return train_, test_

        return X_train, X_test


PREPROCESSOR: Preprocessing = Preprocessing.NOTHING


def __split_features(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X = features[:, :-1]
    y = features[:, -1]
    return X, y


def test(features: np.ndarray, clf, random_state: int = 0, activities: np.ndarray = None, scale: bool = True,
         with_previous_class_feature: bool = False) -> float:
    if not with_previous_class_feature:
        return __test(features, clf, random_state, activities, scale)
    else:
        return __test_with_previous_class_feature(features, clf, activities, scale)


def __test(features: np.ndarray, clf, random_state: int = 0, activities: np.ndarray = None,
           scale: bool = True) -> float:
    X, y = __split_features(features)
    scores = []

    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Normalization
        # norm = Normalizer()
        # X_train = norm.fit_transform(X_train)
        # X_test = norm.transform(X_test)

        X_train, X_test = PREPROCESSOR.preprocess(X_train, X_test)
        # if scale:
        #     sc = StandardScaler()
        #     X_train = sc.fit_transform(X_train)
        #     X_test = sc.transform(X_test)

        clf.fit(X_train, y_train)
        predict = clf.predict(X_test)

        # print("Report:\n", classification_report(y_test, predict, target_names=activities))
        # print("Confusion matrix:\n", confusion_matrix(y_test, predict), end="\n\n")
        acc_score = accuracy_score(y_test, predict)
        scores.append(acc_score)

    return statistics.mean(scores) * 100


def __test_with_previous_class_feature(features: np.ndarray, clf, activities: np.ndarray = None,
                                       scale: bool = True) -> float:
    X, y = __split_features(features)
    scores_overall = []

    kf = KFold(n_splits=5, shuffle=False)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if scale:
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
        clf.fit(X_train, y_train)

        is_first = True
        predictions_fold = []

        vector: np.ndarray
        for vector in X_test:
            if is_first:
                counts = np.bincount(y_train)
                last_class = np.argmax(counts)
                is_first = False
            else:
                last_class = predictions_fold[-1]

            vector[-1] = last_class

            vector = vector.reshape(1, -1)
            if scale:
                vector = sc.transform(vector)
            predict = clf.predict(vector)
            predictions_fold.append(predict)

        # print("Report:\n", classification_report(y_test, predict, target_names=activities))
        # print("Confusion matrix:\n", confusion_matrix(y_test, predict), end="\n\n")
        acc_score = accuracy_score(y_test, predictions_fold)
        scores_overall.append(acc_score)
        # print("score fold:", acc_score)

    return statistics.mean(scores_overall) * 100
