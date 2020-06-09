from typing import Tuple

import numpy as np
from src import data_file_handling as fh, feature_extraction as fex
from src.classifiers import SVM

from src.datasets.Dataset import Dataset
import src.testing.window_size_tester as wst
from src.datasets.Kyoto3 import Kyoto3

path: str = 'kyoto3/report_5/'
# path: str = 'kyoto1/no_preprocessing/'
# path: str = 'kyoto1/normalizer_l1/'
# path: str = 'kyoto1/normalizer_l2/'
# path: str = 'kyoto1/normalizer_max/'
# path: str = 'kyoto1/standard_scaler/'
# path: str = 'kyoto1/robust_scaler/'
# path: str = 'kyoto1/min_max_scaler/'
# path: str = 'kyoto1/max_abs_scaler/'
# save: bool = True
save: bool = False


def test_default_SVC(windows_size: int = 30, with_previous_class_feature: bool = False):
    features = __get_features(windows_size, with_previous_class_feature)
    print(SVM.test_default_SVC(features, with_previous_class_feature=with_previous_class_feature))


def test_all(windows_size: int = 30, with_previous_class_feature: bool = False):
    test_variable_window_sizes(windows_size, with_previous_class_feature)
    test_kernels(windows_size)
    test_c_gamma_default_parameters(windows_size)
    test_c_gamma_parameters(windows_size)
    test_best_SVC(windows_size, with_previous_class_feature)


def test_variable_window_sizes(window_sizes: list = None, with_previous_class_feature: bool = False):
    if window_sizes is None:
        window_sizes = [5, 7, 10, 12, 15, 17, 19, 22, 25, 27, 30, 32, 35, 37, 40]
        # window_sizes = [5, 12, 19, 30, 40]
    data_arrays, sensors = __get_data_arrays()

    # "window_size_testing-first_normalized.txt"
    fname = path + "window_size_testing-PREV_CLASS_VIA_PREDICT_PROBA-NO_SCALE.txt"
    # fname = path + "window_size_testing-CATEG_ENCODING-PREV_CLASS_FEAT-STANDARD.txt" if not with_previous_class_feature else \
    #     path + "window_size_testing-with_PCF.txt"
    wst.test_variable_window_sizes(data_arrays, sensors, window_sizes, with_previous_class_feature,
                                   fname if save else None)


def test_kernels(windows_size: int = 30):
    features = __get_features(windows_size)
    SVM.test_kernels(features, path + "kernel_testing_ws" + str(windows_size) + ".txt" if save else None)


def test_c_gamma_default_parameters(windows_size: int = 30):
    features = __get_features(windows_size)
    SVM.test_c_gamma_default_parameters(features, path + "c_gamma_default_testing_ws" +
                                                  str(windows_size) + ".txt" if save else None)


def test_c_gamma_parameters(windows_size: int = 30):
    features = __get_features(windows_size)
    SVM.test_c_gamma_parameters(features, path + "c_gamma_testing_ws" + str(windows_size) + ".txt" if save else None)


def test_best_SVC(windows_size: int = 30, with_previous_class_feature: bool = False):
    features = __get_features(windows_size, with_previous_class_feature)
    SVM.test_best_SVC(features, __get_activities(), with_previous_class_feature=with_previous_class_feature)


def __get_data_arrays() -> Tuple[list, list]:
    dataset: Dataset = Kyoto3()
    return fh.get_data_arrays_from_directory_kyoto3(dataset), dataset.sensors


def __get_features(windows_size: int, with_previous_class_feature: bool = False) -> np.ndarray:
    data_arrays, sensors = __get_data_arrays()
    return fex.extract_features_from_arrays(data_arrays, windows_size, sensors, with_previous_class_feature)


def __get_activities() -> np.ndarray:
    return np.array(Kyoto3().activities)
