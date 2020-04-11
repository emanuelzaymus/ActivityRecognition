import numpy as np
from sklearn.svm import SVC, LinearSVC
from typing import List
import classifier


def test_default_SVC(features: np.ndarray) -> float:
    clf = SVC()
    return classifier.test(features, clf)


def test_kernels(features: np.ndarray):
    kernels = ('linear', 'poly', 'rbf', 'sigmoid')
    file_to_save: np.ndarray = np.empty((len(kernels), 2), dtype='object')

    for i in range(len(kernels)):
        print("Kernel:", kernels[i])
        file_to_save[i, 0] = kernels[i]
        clf = SVC(kernel=kernels[i])

        accuracy = classifier.test(features, clf)
        print("Accuracy:", accuracy)
        file_to_save[i, 1] = accuracy

    np.savetxt("results/kernel_testing.txt", file_to_save, delimiter='\t', fmt="%s")


def test_c_gamma_default_parameters(features: np.ndarray):
    c_regulations = [.25, .5, 1, 2, 5, 7, 10, 13, 15, 17, 20, 30, 50, 75, 100, 150, 200, 500, 1000]
    gammas = ['scale', 'auto']

    file_to_save: np.ndarray = np.empty((len(c_regulations) * len(gammas), 3), dtype=object)
    __test_c_gamma_parameters(features, c_regulations, gammas, file_to_save)
    np.savetxt("results/c_gamma_default_testing.txt", file_to_save, delimiter='\t', fmt="%s")


def test_c_gamma_parameters(features: np.ndarray):
    c_regulations = [1, 2, 5, 7, 10, 13, 15, 17, 20, 30, 50, 75, 100, 150, 200]
    gammas = [.1, .15, .2, .25, .3, .35, .4, .45]

    file_to_save: np.ndarray = np.empty((len(c_regulations) * len(gammas), 3), dtype=object)
    __test_c_gamma_parameters(features, c_regulations, gammas, file_to_save)
    np.savetxt("results/c_gamma_testing.txt", file_to_save, delimiter='\t', fmt="%s")


def test_best_SVC(features: np.ndarray, activities: np.ndarray = None):
    clf = SVC(kernel='rbf', C=20, gamma=0.2)
    # clf = SVC(kernel='rbf', C=13, gamma=0.3)
    print("TOTAL accuracy score:", classifier.test(features, clf, activities=activities))


def test_best_SVC_with_PCF(features: np.ndarray, activities: np.ndarray = None):
    clf = SVC(kernel='rbf', C=20, gamma=0.2)
    # clf = SVC(kernel='rbf', C=13, gamma=0.3)
    print("TOTAL accuracy score:", classifier.test_with_previous_class_feature(features, clf, activities=activities))


def __test_c_gamma_parameters(features: np.ndarray, c_regulations: List[float], gammas: List,
                              file_to_save: np.ndarray):
    i = 0
    for c in c_regulations:
        for g in gammas:
            print("C:", c, " Gamma:", g)
            file_to_save[i, 0] = c
            file_to_save[i, 1] = g

            clf = SVC(kernel='rbf', C=c, gamma=g)

            accuracy = classifier.test(features, clf)
            print("Accuracy:", accuracy)
            file_to_save[i, 2] = accuracy
            i += 1
