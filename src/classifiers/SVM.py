import numpy as np
from sklearn.svm import SVC
from typing import List
from src.classifiers import classifier


def test_default_SVC(features: np.ndarray, with_previous_class_feature: bool = False) -> float:
    clf = SVC()
    return classifier.test(features, clf, with_previous_class_feature=with_previous_class_feature)


def test_kernels(features: np.ndarray, fname_to_save: str = None):
    kernels = ('linear', 'poly', 'rbf', 'sigmoid')
    file_to_save: np.ndarray = np.empty((len(kernels), 2), dtype='object')

    for i in range(len(kernels)):
        print("Kernel:", kernels[i])
        file_to_save[i, 0] = kernels[i]
        clf = SVC(kernel=kernels[i])

        accuracy = classifier.test(features, clf)
        print("Accuracy:", accuracy)
        file_to_save[i, 1] = accuracy

    if fname_to_save is not None:
        np.savetxt("results/" + fname_to_save, file_to_save, delimiter='\t', fmt="%s")


def test_c_gamma_default_parameters(features: np.ndarray, fname_to_save: str = None):
    c_regulations = [.25, .5, 1, 2, 5, 7, 10, 13, 15, 17, 20, 30, 50, 75, 100, 150, 200, 500, 1000]
    gammas = ['scale', 'auto']

    file_to_save: np.ndarray = np.empty((len(c_regulations) * len(gammas), 3), dtype=object)
    __test_c_gamma_parameters(features, c_regulations, gammas, file_to_save)
    if fname_to_save is not None:
        np.savetxt("results/" + fname_to_save, file_to_save, delimiter='\t', fmt="%s")


def test_c_gamma_parameters(features: np.ndarray, fname_to_save: str = None):
    c_regulations = [1, 2, 5, 7, 10, 13, 15, 17, 20, 30, 50, 75, 100, 150, 200]
    gammas = [.1, .15, .2, .25, .3, .35, .4, .45]

    file_to_save: np.ndarray = np.empty((len(c_regulations) * len(gammas), 3), dtype=object)
    __test_c_gamma_parameters(features, c_regulations, gammas, file_to_save)
    if fname_to_save is not None:
        np.savetxt("results/" + fname_to_save, file_to_save, delimiter='\t', fmt="%s")


def test_best_SVC(features: np.ndarray, activities: np.ndarray = None, with_previous_class_feature: bool = False):
    clf = SVC(kernel='rbf', C=20, gamma=0.2)
    # clf = SVC(kernel='rbf', C=13, gamma=0.3)
    print("TOTAL accuracy score:", classifier.test(features, clf, activities=activities,
                                                   with_previous_class_feature=with_previous_class_feature))


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
