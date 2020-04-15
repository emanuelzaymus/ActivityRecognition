from typing import Tuple

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


def __split_features(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X = features[:, :-1]
    y = features[:, -1]
    return X, y


def test(features: np.ndarray, clf, random_state: int = 0, activities: np.ndarray = None, scale: bool = True) -> float:
    X, y = __split_features(features)
    scores: np.ndarray = np.array([])

    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if scale:
            sc = StandardScaler()
            X_train = sc.fit_transform(X_train)
            X_test = sc.transform(X_test)

        clf.fit(X_train, y_train)
        predict = clf.predict(X_test)

        # print("Report:\n", classification_report(y_test, predict, target_names=activities))
        # print("Confusion matrix:\n", confusion_matrix(y_test, predict), end="\n\n")
        acc_score = accuracy_score(y_test, predict)
        scores = np.append(scores, acc_score)

    return np.mean(scores) * 100


def test_with_previous_class_feature(features: np.ndarray, clf, activities: np.ndarray = None) -> float:
    X, y = __split_features(features)
    scores_overall: np.ndarray = np.array([])

    kf = KFold(n_splits=5, shuffle=False)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

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
            vector = sc.transform(vector)
            predict = clf.predict(vector)
            predictions_fold.append(predict)

        # print("Report:\n", classification_report(y_test, predict, target_names=activities))
        # print("Confusion matrix:\n", confusion_matrix(y_test, predict), end="\n\n")
        acc_score = accuracy_score(y_test, predictions_fold)
        scores_overall = np.append(scores_overall, acc_score)
        print("score fold:", acc_score)

    return np.mean(scores_overall) * 100
