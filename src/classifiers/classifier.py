import statistics
from enum import Enum
from typing import Tuple

import numpy as np

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, Normalizer, RobustScaler, MinMaxScaler, MaxAbsScaler
from sklearn.svm import SVC


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

        if self is not Preprocessing.NOTHING:
            train_ = self.value.fit_transform(X_train)
            test_ = self.value.transform(X_test)
            return train_, test_

        return X_train, X_test

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        if self is not Preprocessing.NOTHING:
            return self.value.fit_transform(X_train)

        return X_train

    def transform(self, feat_vector: np.ndarray) -> np.ndarray:
        if self is not Preprocessing.NOTHING:
            return self.value.transform(feat_vector)

        return feat_vector


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
        # return __test_with_previous_class_feature(features, clf, activities, scale)
        return __test_with_previous_class_feature_predict_proba(features, clf, activities, scale)
        # return __test_with_previous_class_feature_decision_func(features, clf, activities, scale)


def __test(features: np.ndarray, clf, random_state: int = 0, activities: np.ndarray = None,
           scale: bool = True) -> float:
    X, y = __split_features(features)
    scores = []

    # TODO: shuffle=True
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    # kf = KFold(n_splits=5, shuffle=False)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train, X_test = PREPROCESSOR.preprocess(X_train, X_test)
        clf.fit(X_train, y_train)
        predict = clf.predict(X_test)

        # print("Report:\n", classification_report(y_test, predict, target_names=activities))
        # print("Confusion matrix:\n", confusion_matrix(y_test, predict), end="\n\n")
        acc_score = accuracy_score(y_test, predict)
        scores.append(acc_score)

    return statistics.mean(scores) * 100


def __test2(features: np.ndarray, clf, random_state: int = 0, activities: np.ndarray = None,
            scale: bool = True) -> float:
    X, y = __split_features(features)
    scores_overall = []

    kf = KFold(n_splits=5, shuffle=False)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)

        clf.fit(X_train, y_train)

        predictions_fold = []

        vector: np.ndarray
        for vector in X_test:
            vector = vector.reshape(1, -1)
            vector = sc.transform(vector)
            predict = clf.predict(vector)
            predictions_fold.append(predict)

        acc_score = accuracy_score(y_test, predictions_fold)
        scores_overall.append(acc_score)
        # print("score fold:", acc_score)

    return statistics.mean(scores_overall) * 100


def __test_with_previous_class_feature(features: np.ndarray, clf, activities: np.ndarray = None,
                                       scale: bool = True) -> float:
    X, y = __split_features(features)
    scores_overall = []

    kf = KFold(n_splits=5, shuffle=False)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train = PREPROCESSOR.fit_transform(X_train)
        clf.fit(X_train, y_train)

        is_first = True
        predictions_fold = []

        vector: np.ndarray
        for vector in X_test:
            if is_first:
                last_class = statistics.mode(y_train)
                is_first = False
            else:
                last_class = predictions_fold[-1]

            vector[-5:] = 0
            index: int = int(-last_class - 1)
            vector[index] = 1

            vector = vector.reshape(1, -1)
            vector = PREPROCESSOR.transform(vector)
            predict = clf.predict(vector)
            predictions_fold.append(predict)

        # print("Report:\n", classification_report(y_test, predictions_fold, target_names=activities))
        # print("Confusion matrix:\n", confusion_matrix(y_test, predictions_fold), end="\n\n")
        acc_score = accuracy_score(y_test, predictions_fold)
        scores_overall.append(acc_score)
        # print("score fold:", acc_score)

    return statistics.mean(scores_overall) * 100


def __test_with_previous_class_feature_predict_proba(features: np.ndarray, clf: SVC, activities: np.ndarray = None,
                                                     scale: bool = True) -> float:
    X, y = __split_features(features)
    scores_overall = []

    kf = KFold(n_splits=5, shuffle=False)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train = PREPROCESSOR.fit_transform(X_train)
        clf.fit(X_train, y_train)

        predictions_fold = []

        # print('classes:', clf.classes_)

        for i in range(X_test.shape[0]):
            vector: np.ndarray = X_test[i]
            if i == 0:
                vector[-5:] = 0.2
            else:
                last: np.ndarray = clf.predict_proba([X_test[i - 1]])
                # print('last:', last, end=' ')
                vector[-5:] = last

            # Test
            vector = vector.reshape(1, -1)
            vector = PREPROCESSOR.transform(vector)
            predict = clf.predict(vector)
            # print('predict:', predict, 'actual:', y_test[i])
            predictions_fold.append(predict)

        # print("Report:\n", classification_report(y_test, predictions_fold, target_names=activities))
        # print("Confusion matrix:\n", confusion_matrix(y_test, predictions_fold), end="\n\n")
        acc_score = accuracy_score(y_test, predictions_fold)
        scores_overall.append(acc_score)
        # print("score fold:", acc_score)

    return statistics.mean(scores_overall) * 100


def __test_with_previous_class_feature_decision_func(features: np.ndarray, clf: SVC, activities: np.ndarray = None,
                                                     scale: bool = True) -> float:
    X, y = __split_features(features)
    scores_overall = []

    maximum = -9999999999

    kf = KFold(n_splits=5, shuffle=False)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # np.place(X_train[:, -5:], X_train[:, -5:] == 1, 5)

        X_train = PREPROCESSOR.fit_transform(X_train)
        clf.fit(X_train, y_train)

        predictions_fold = []

        # print('classes:', clf.classes_)

        for i in range(X_test.shape[0]):
            vector: np.ndarray = X_test[i]
            if i == 0:
                vector[-5:] = 0.2
            else:
                prev: np.ndarray = np.array([X_train[i - 1]])

                # print(prev)
                # print(prev.astype(int))
                last: np.ndarray = clf.decision_function(prev)
                # print('last:', last, end=' ')
                vector[-5:] = last

            if maximum < np.amax(vector[-5:]):
                maximum = np.amax(vector[-5:])

            # Test
            vector = vector.reshape(1, -1)
            vector = PREPROCESSOR.transform(vector)
            # print('transformed:', vector[0, -5:], end=' ')
            predict = clf.predict(vector)
            # print('predict:', predict, 'actual:', y_test[i])
            predictions_fold.append(predict)

        # print("Report:\n", classification_report(y_test, predictions_fold, target_names=activities))
        # print("Confusion matrix:\n", confusion_matrix(y_test, predictions_fold), end="\n\n")
        acc_score = accuracy_score(y_test, predictions_fold)
        scores_overall.append(acc_score)
        # print("score fold:", acc_score)

    print('MAXIMUM:', maximum)
    return statistics.mean(scores_overall) * 100
