import time
from typing import List

import numpy as np
from sklearn.svm import SVC

import src.feature_extraction as fex
from src.classifiers import classifier
from src.classifiers.Kernel import Kernel


def test_default_svm(features: np.ndarray, activities: np.ndarray = None, kernel: Kernel = Kernel.RBF) -> float:
    clf = SVC(kernel=kernel.value)
    return classifier.test(features, clf, activities=activities)


def test_variable_window_sizes(data_arrays: List[np.ndarray], sensors: List[str], window_sizes: List[int],
                               f_name_to_save: str = None, with_time_duration: bool = False):
    file_to_save: np.ndarray = np.zeros((len(window_sizes), 2 if not with_time_duration else 3))

    for i in range(len(window_sizes)):
        print("Window size:", window_sizes[i])
        file_to_save[i, 0] = window_sizes[i]
        features = fex.extract_features_from_arrays(data_arrays, window_sizes[i], sensors)

        start_time = time.time()
        accuracy: float = test_default_svm(features)
        print("Accuracy:", accuracy)
        file_to_save[i, 1] = accuracy

        if with_time_duration:
            file_to_save[i, 2] = time.time() - start_time
            print("Time:", file_to_save[i, 2])

    if f_name_to_save is not None:
        np.savetxt("results/" + f_name_to_save, file_to_save, delimiter='\t', fmt="%.4f")


def test_kernels(features: np.ndarray, f_name_to_save: str = None):
    kernels = ('linear', 'poly', 'rbf', 'sigmoid')
    file_to_save: np.ndarray = np.empty((len(kernels), 2), dtype='object')

    for i in range(len(kernels)):
        print("Kernel:", kernels[i])
        file_to_save[i, 0] = kernels[i]

        clf = SVC(kernel=kernels[i])
        file_to_save[i, 1] = classifier.test(features, clf)
        print("Accuracy:", file_to_save[i, 1])

    if f_name_to_save is not None:
        np.savetxt("results/" + f_name_to_save, file_to_save, delimiter='\t', fmt="%s")


def test_kernel_c_gamma_parameters(features: np.ndarray, kernel: Kernel, c_regulations: List[int], gammas: List[float],
                                   f_name_to_save: str = None, with_time_duration: bool = False):
    file_to_save = np.empty((len(c_regulations) * len(gammas), 3 if not with_time_duration else 4), dtype=object)

    __test_c_gamma_parameters(kernel.value, features, c_regulations, gammas, file_to_save, with_time_duration)

    if f_name_to_save is not None:
        np.savetxt("results/" + f_name_to_save, file_to_save, delimiter='\t', fmt="%s")


def __test_c_gamma_parameters(kernel: str, features: np.ndarray, c_regulations: List[float], gammas: List,
                              file_to_save: np.ndarray, with_time_duration: bool = False):
    i = 0
    for c in c_regulations:
        for g in gammas:
            print("C:", c, " Gamma:", g)
            file_to_save[i, 0] = c
            file_to_save[i, 1] = g

            start_time = time.time()
            clf = SVC(kernel=kernel, C=c, gamma=g)
            file_to_save[i, 2] = classifier.test(features, clf)
            print("Accuracy:", file_to_save[i, 2])

            if with_time_duration:
                file_to_save[i, 3] = time.time() - start_time
                print("Time:", file_to_save[i, 3])

            i += 1


def test_pca(features: np.ndarray, pca_n_components_list: List[float], f_name_to_save: str = None):
    file_to_save: np.ndarray = np.empty((len(pca_n_components_list), 3), dtype=object)

    for i in range(len(pca_n_components_list)):
        file_to_save[i, 0] = pca_n_components_list[i]
        print("PCA n_components:", file_to_save[i, 0])

        start_time = time.time()
        file_to_save[i, 1] = test_default_svm(features)
        print("Accuracy:", file_to_save[i, 1])

        file_to_save[i, 2] = time.time() - start_time
        print("Time:", (file_to_save[i, 2]))

    if f_name_to_save is not None:
        np.savetxt("results/" + f_name_to_save, file_to_save, delimiter='\t', fmt="%s")


def test_best_svm(features: np.ndarray, activities: np.ndarray, kernel: Kernel, c: int, gamma: float):
    clf = SVC(kernel=kernel.value, C=c, gamma=gamma)
    print("Accuracy score:", classifier.test(features, clf, activities=activities))
