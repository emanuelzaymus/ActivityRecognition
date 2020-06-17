import time
from typing import List

import numpy as np
from sklearn.svm import SVC

import src.feature_extraction as fex
from src.classifiers import classifier
from src.testing import TestParameters as Params


def test_default_svm(features: np.ndarray, activities: np.ndarray = None,
                     with_previous_class_feature: bool = False) -> float:
    clf = SVC(probability=with_previous_class_feature, break_ties=with_previous_class_feature)
    return classifier.test(features, clf, activities=activities,
                           with_previous_class_feature=with_previous_class_feature)


def test_variable_window_sizes(data_arrays: List[np.ndarray], sensors: List[str], window_sizes: List[int],
                               with_previous_class_feature: bool = False, f_name_to_save: str = None,
                               with_time_duration: bool = False):
    file_to_save: np.ndarray = np.zeros((len(window_sizes), 2 if not with_time_duration else 3))

    for i in range(len(window_sizes)):
        print("Window size:", window_sizes[i])
        file_to_save[i, 0] = window_sizes[i]
        features = fex.extract_features_from_arrays(data_arrays, window_sizes[i], sensors,
                                                    with_previous_class_feature=with_previous_class_feature)

        start_time = time.time()
        accuracy: float = test_default_svm(features, with_previous_class_feature=with_previous_class_feature)
        print("Accuracy:", accuracy, '\n')
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

        accuracy = classifier.test(features, clf)
        print("Accuracy:", accuracy)
        file_to_save[i, 1] = accuracy

    if f_name_to_save is not None:
        np.savetxt("results/" + f_name_to_save, file_to_save, delimiter='\t', fmt="%s")


def test_rbf_c_gamma_default_parameters(features: np.ndarray, f_name_to_save: str = None):
    c_regulations = [.25, .5, 1, 2, 5, 7, 10, 13, 15, 17, 20, 30, 50, 75, 100, 150, 200, 500, 1000]
    gammas = ['scale', 'auto']

    file_to_save: np.ndarray = np.empty((len(c_regulations) * len(gammas), 3), dtype=object)
    __test_c_gamma_parameters('rbf', features, c_regulations, gammas, file_to_save)
    if f_name_to_save is not None:
        np.savetxt("results/" + f_name_to_save, file_to_save, delimiter='\t', fmt="%s")


def test_rbf_c_gamma_parameters(features: np.ndarray, f_name_to_save: str = None, with_time_duration: bool = True):
    c_regulations = Params.C_REGULATIONS
    gammas = Params.GAMMAS

    file_to_save: np.ndarray = np.empty((len(c_regulations) * len(gammas), 3 if not with_time_duration else 4),
                                        dtype=object)
    __test_c_gamma_parameters('rbf', features, c_regulations, gammas, file_to_save,
                              with_time_duration=with_time_duration)
    if f_name_to_save is not None:
        np.savetxt("results/" + f_name_to_save, file_to_save, delimiter='\t', fmt="%s")


def test_best_svm(features: np.ndarray, activities: np.ndarray = None, with_previous_class_feature: bool = False):
    clf = SVC(kernel=Params.KERNEL.value, C=Params.C, gamma=Params.GAMMA)
    print("TOTAL accuracy score:", classifier.test(features, clf, activities=activities,
                                                   with_previous_class_feature=with_previous_class_feature))


def __test_c_gamma_parameters(kernel: str, features: np.ndarray, c_regulations: List[float], gammas: List,
                              file_to_save: np.ndarray, with_time_duration: bool = False):
    i = 0
    for c in c_regulations:
        for g in gammas:
            # if not (c == 13 and g == 0.2):
            #     print("!!")
            #     continue

            print("C:", c, " Gamma:", g)
            file_to_save[i, 0] = c
            file_to_save[i, 1] = g

            start_time = time.time()
            clf = SVC(kernel=kernel, C=c, gamma=g)

            accuracy = classifier.test(features, clf)
            print("Accuracy:", accuracy)
            file_to_save[i, 2] = accuracy

            if with_time_duration:
                file_to_save[i, 3] = time.time() - start_time
                print(file_to_save[i, 3])

            i += 1


def test_poly_defaults(features: np.ndarray, activities: np.ndarray, with_previous_class_feature: bool = False):
    clf = SVC(kernel='poly')
    print("TOTAL accuracy score:", classifier.test(features, clf, activities=activities,
                                                   with_previous_class_feature=with_previous_class_feature))


def test_poly_c_gamma_parameters(features: np.ndarray, f_name_to_save: str = None, with_time_duration: bool = False):
    c_regulations = Params.C_REGULATIONS
    gammas = Params.GAMMAS

    file_to_save: np.ndarray = np.empty((len(c_regulations) * len(gammas), 3 if not with_time_duration else 4),
                                        dtype=object)
    __test_c_gamma_parameters('poly', features, c_regulations, gammas, file_to_save,
                              with_time_duration=with_time_duration)
    if f_name_to_save is not None:
        np.savetxt("results/" + f_name_to_save, file_to_save, delimiter='\t', fmt="%s")


def test_pca(features: np.ndarray, f_name_to_save: str = None):
    file_to_save: np.ndarray = np.empty((len(Params.PCA_N_COMPONENTS_LIST), 3), dtype=object)

    for i in range(len(Params.PCA_N_COMPONENTS_LIST)):
        Params.PCA_N_COMPONENTS = Params.PCA_N_COMPONENTS_LIST[i]
        print(Params.PCA_N_COMPONENTS)
        file_to_save[i, 0] = Params.PCA_N_COMPONENTS

        start_time = time.time()
        file_to_save[i, 1] = test_default_svm(features)
        print("Accuracy:", file_to_save[i, 1])

        file_to_save[i, 2] = time.time() - start_time
        print('Execution time: %s s' % (file_to_save[i, 2]))

    if f_name_to_save is not None:
        np.savetxt("results/" + f_name_to_save, file_to_save, delimiter='\t', fmt="%s")
