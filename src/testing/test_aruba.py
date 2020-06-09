from typing import Tuple

import numpy as np
from src import data_file_handling as fh, feature_extraction as fex
from src.classifiers import SVM
from src.datasets.Arbua import Aruba

import src.testing.window_size_tester as wst

path: str = 'aruba/report_5/'
# save: bool = True


save: bool = False


def test_default_SVC(windows_size: int = 30, with_previous_class_feature: bool = False):
    features, activities = __get_features_activities(windows_size, with_previous_class_feature)
    print(
        SVM.test_default_SVC(features, activities=activities, with_previous_class_feature=with_previous_class_feature))


def test_all(windows_size: int = 30, with_previous_class_feature: bool = False):
    test_variable_window_sizes(windows_size, with_previous_class_feature)
    test_kernels(windows_size)
    test_c_gamma_default_parameters(windows_size)
    test_c_gamma_parameters(windows_size)
    test_best_SVC(windows_size, with_previous_class_feature)


def test_variable_window_sizes(window_sizes: list = None, with_previous_class_feature: bool = False, what: str = None):
    if window_sizes is None:
        window_sizes = [5, 7, 10, 12, 15, 17, 19, 22, 25, 27, 30, 32, 35, 37, 40]

    # "window_size_testing-first_normalized.txt"
    fname = path + "window_size_testing-" + what + ".txt"
    # fname = path + "window_size_testing-CATEG_ENCODING-PREV_CLASS_FEAT-STANDARD.txt" if not with_previous_class_feature else \
    #     path + "window_size_testing-with_PCF.txt"
    wst.test_variable_window_sizes_aruba(__get_data_array(), window_sizes, with_previous_class_feature,
                                         fname if save else None)


def test_kernels(windows_size: int = 30):
    features, a = __get_features_activities(windows_size)
    SVM.test_kernels(features, path + "kernel_testing_ws" + str(windows_size) + ".txt" if save else None)


def test_c_gamma_default_parameters(windows_size: int = 30):
    features, a = __get_features_activities(windows_size)
    SVM.test_c_gamma_default_parameters(features, path + "c_gamma_default_testing_ws" +
                                                  str(windows_size) + ".txt" if save else None)


def test_c_gamma_parameters(windows_size: int = 30):
    features, a = __get_features_activities(windows_size)
    SVM.test_c_gamma_parameters(features,
                                path + "c_gamma_testing_ws" + str(windows_size) + ".txt" if save else None)


def test_best_SVC(windows_size: int = 30, with_previous_class_feature: bool = False):
    features, activities = __get_features_activities(windows_size)
    SVM.test_best_SVC(features, activities, with_previous_class_feature=with_previous_class_feature)


def __get_features_activities(windows_size: int, with_previous_class_feature: bool = False) -> Tuple[
    np.ndarray, np.ndarray]:
    return fex.extract_features(__get_data_array(), windows_size,
                                with_previous_class_feature=with_previous_class_feature)


def __get_data_array() -> np.ndarray:
    return fh.get_data_array(Aruba().file)
