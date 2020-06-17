import statistics
from typing import Tuple

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.svm import SVC

import src.testing.TestParameters as Params
from src.classifiers import PCA


def __split_features(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X = features[:, :-1]
    y = features[:, -1]
    return X, y


def test(features: np.ndarray, clf, random_state: int = Params.RANDOM_STATE, activities: np.ndarray = None,
         with_previous_class_feature: bool = False) -> float:
    if not with_previous_class_feature:
        return __test(features, clf, random_state, activities)
    else:
        return __test_with_previous_class_feature_predict_proba(features, clf, activities)
        # return __test_with_previous_class_feature_decision_func(features, clf, activities, scale)


# noinspection PyPep8Naming
def __test(features: np.ndarray, clf, random_state: int, activities: np.ndarray = None) -> float:
    X, y = __split_features(features)
    scores = []

    # kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    # for train_index, test_index in kf.split(X):
    #     X_train, X_test = X[train_index], X[test_index]
    #     y_train, y_test = y[train_index], y[test_index]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    print("splitted")

    X_train, X_test = Params.PREPROCESSOR.preprocess(X_train, X_test)
    print("scaled")
    X_train, X_test = PCA.decompose(X_train, X_test)

    clf.fit(X_train, y_train)
    print("trained")
    predict = clf.predict(X_test)
    print("tested")

    # print("Report:\n", classification_report(y_test, predict))  # , target_names=activities))
    # print("Confusion matrix:\n", confusion_matrix(y_test, predict), end="\n\n")
    acc_score = accuracy_score(y_test, predict)
    scores.append(acc_score)

    return statistics.mean(scores) * 100


# noinspection PyPep8Naming
def __test_with_previous_class_feature(features: np.ndarray, clf, activities: np.ndarray = None) -> float:
    X, y = __split_features(features)
    scores_overall = []

    kf = KFold(n_splits=5, shuffle=False)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train = Params.PREPROCESSOR.fit_transform(X_train)
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
            vector = Params.PREPROCESSOR.transform(vector)
            predict = clf.predict(vector)
            predictions_fold.append(predict)

        # print("Report:\n", classification_report(y_test, predictions_fold, target_names=activities))
        # print("Confusion matrix:\n", confusion_matrix(y_test, predictions_fold), end="\n\n")
        acc_score = accuracy_score(y_test, predictions_fold)
        scores_overall.append(acc_score)
        # print("score fold:", acc_score)

    return statistics.mean(scores_overall) * 100


# noinspection PyPep8Naming
def __test_with_previous_class_feature_predict_proba(features: np.ndarray, clf: SVC,
                                                     activities: np.ndarray = None) -> float:
    X, y = __split_features(features)
    scores_overall = []

    kf = KFold(n_splits=5, shuffle=False)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train = Params.PREPROCESSOR.fit_transform(X_train)
        clf.fit(X_train, y_train)

        predictions_fold = []

        print('classes:', clf.classes_)

        for i in range(X_test.shape[0]):
            vector: np.ndarray = X_test[i]
            if i == 0:
                vector[-5:] = 1 / 5
            else:
                last: np.ndarray = clf.predict_proba([X_test[i - 1]])
                # print('last:', last, end=' ')
                vector[-5:] = last

            # Test
            vector = vector.reshape(1, -1)
            vector = Params.PREPROCESSOR.transform(vector)
            predict = clf.predict(vector)
            # pp = clf.predict_proba(vector)
            # print('predict:', predict, 'predict_proba:', pp, 'actual:', y_test[i])
            predictions_fold.append(predict)

        # print(np.unique(y_test))

        # print("Report:\n", classification_report(y_test, predictions_fold))  # , target_names=activities))
        # print("Confusion matrix:\n", confusion_matrix(y_test, predictions_fold), end="\n\n")
        acc_score = accuracy_score(y_test, predictions_fold)
        scores_overall.append(acc_score)
        print("score fold:", acc_score)

    return statistics.mean(scores_overall) * 100
