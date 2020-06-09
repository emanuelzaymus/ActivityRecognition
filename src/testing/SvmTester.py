from src.classifiers import SVM
from src.datasets.Dataset import Dataset
import src.testing.window_size_tester as wst


class SvmTester:

    def __init__(self, dataset: Dataset, save: bool = False, save_to_file: str = None):
        self.dataset = dataset
        self.save = save
        self.save_to_file = save_to_file

    def __get_f_name(self, s: str, n: int = None):
        if self.save and self.save_to_file is not None:
            return self.save_to_file + s + (str(n) if n is not None else '') + ".txt"
        return None

    def test_default_svm(self, windows_size: int = 30, with_previous_class_feature: bool = False):
        features = self.dataset.get_features(windows_size, with_previous_class_feature)
        print(SVM.test_default_SVC(features, with_previous_class_feature=with_previous_class_feature))

    def test_variable_window_sizes(self, window_sizes: list = None, with_previous_class_feature: bool = False):
        if window_sizes is None:
            window_sizes = [5, 7, 10, 12, 15, 17, 19, 22, 25, 27, 30, 32, 35, 37, 40]  # [5, 12, 19, 30, 40]
        data_arrays, sensors = self.dataset.get_data_arrays()

        f_name = self.__get_f_name("window_size_testing-PREV_CLASS_VIA_PREDICT_PROBA-NO_SCALE.txt")
        wst.test_variable_window_sizes(data_arrays, sensors, window_sizes, with_previous_class_feature, f_name)

    def test_kernels(self, windows_size: int = 30):
        features = self.dataset.get_features(windows_size)
        SVM.test_kernels(features, self.__get_f_name("kernel_testing_ws", windows_size))

    def test_c_gamma_default_parameters(self, windows_size: int = 30):
        features = self.dataset.get_features(windows_size)
        SVM.test_c_gamma_default_parameters(features, self.__get_f_name("c_gamma_default_testing_ws", windows_size))

    def test_c_gamma_parameters(self, windows_size: int = 30):
        features = self.dataset.get_features(windows_size)
        SVM.test_c_gamma_parameters(features, self.__get_f_name("c_gamma_testing_ws", windows_size))

    def test_best_svm(self, windows_size: int = 30, with_previous_class_feature: bool = False):
        features = self.dataset.get_features(windows_size, with_previous_class_feature)
        SVM.test_best_SVC(features, self.dataset.get_activities(),
                          with_previous_class_feature=with_previous_class_feature)
