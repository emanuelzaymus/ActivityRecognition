from typing import List

import src.testing.TestParameters as Params
from src.classifiers import SVM
from src.datasets.Dataset import Dataset


class SvmTester:

    def __init__(self, dataset: Dataset, save_to_file: str = None):
        self.dataset = dataset
        self.save_to_file = save_to_file

    def __get_f_name(self, s: str, n: int = None):
        if self.save_to_file is not None:
            return self.save_to_file + s + (str(n) if n is not None else '') + ".txt"
        return None

    def test_default_svm(self, windows_size: int = Params.WINDOW_SIZE,
                         with_previous_class_feature: bool = Params.WITH_PREVIOUS_CLASS_FEATURE):
        features = self.dataset.get_features(windows_size, with_previous_class_feature)
        print(SVM.test_default_svm(features, with_previous_class_feature=with_previous_class_feature))

    def test_variable_window_sizes(self, window_sizes: List[int] = None, file_name: str = "window_size_testing",
                                   with_previous_class_feature: bool = False, with_time_duration: bool = False):
        if window_sizes is None:
            window_sizes = Params.WINDOW_SIZES
        data_arrays, sensors = self.dataset.get_data_arrays()
        f_name = self.__get_f_name(file_name)
        SVM.test_variable_window_sizes(data_arrays, sensors, window_sizes, with_previous_class_feature, f_name,
                                       with_time_duration)

    def test_kernels(self, windows_size: int = Params.WINDOW_SIZE):
        features = self.dataset.get_features(windows_size)
        SVM.test_kernels(features, self.__get_f_name("kernel_testing_ws", windows_size))

    def test_rbf_c_gamma_default_parameters(self, windows_size: int = Params.WINDOW_SIZE):
        features = self.dataset.get_features(windows_size)
        SVM.test_rbf_c_gamma_default_parameters(features, self.__get_f_name("c_gamma_default_testing_ws", windows_size))

    def test_rbf_c_gamma_parameters(self, windows_size: int = Params.WINDOW_SIZE, with_time_duration: bool = False):
        features = self.dataset.get_features(windows_size)
        SVM.test_rbf_c_gamma_parameters(features, self.__get_f_name("rbf_c_gamma_testing_ws", windows_size),
                                        with_time_duration=with_time_duration)

    def test_best_svm(self, windows_size: int = Params.WINDOW_SIZE,
                      with_previous_class_feature: bool = Params.WITH_PREVIOUS_CLASS_FEATURE):
        features = self.dataset.get_features(windows_size, with_previous_class_feature)
        print("got features")
        SVM.test_best_svm(features, self.dataset.get_activities(),
                          with_previous_class_feature=with_previous_class_feature)

    def test_poly_defaults(self, windows_size: int = Params.WINDOW_SIZE,
                           with_previous_class_feature: bool = Params.WITH_PREVIOUS_CLASS_FEATURE):
        features = self.dataset.get_features(windows_size, with_previous_class_feature)
        SVM.test_poly_defaults(features, self.dataset.get_activities(),
                               with_previous_class_feature=with_previous_class_feature)

    def test_poly_c_gamma_parameters(self, windows_size: int = Params.WINDOW_SIZE, with_time_duration: bool = False):
        features = self.dataset.get_features(windows_size)
        SVM.test_poly_c_gamma_parameters(features, self.__get_f_name("poly_c_gamma_testing_ws", windows_size),
                                         with_time_duration=with_time_duration)

    def test_pca(self, windows_size: int = Params.WINDOW_SIZE):
        features = self.dataset.get_features(windows_size)
        SVM.test_pca(features, self.__get_f_name("pca_testing_ws", windows_size))
