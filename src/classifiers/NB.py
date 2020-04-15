import numpy as np
from sklearn.naive_bayes import CategoricalNB

from src.classifiers import classifier


def test_CategoricalNB(features: np.ndarray, activities: np.ndarray = None):
    clf = CategoricalNB()
    accuracy = classifier.test(features, clf, activities=activities, scale=False)
    print("CategoricalNB:", accuracy)
