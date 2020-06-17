import src.testing.SvmParameters as Params
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

    def test_default_svm(self):
        features = self.dataset.get_features(Params.WINDOW_SIZE)
        print(SVM.test_default_svm(features, self.dataset.get_activities(), Params.KERNEL))

    def test_variable_window_sizes(self, new_file_name: str = "window_size_testing"):
        data_arrays, sensors = self.dataset.get_data_arrays()
        f_name = self.__get_f_name(new_file_name)
        SVM.test_variable_window_sizes(data_arrays, sensors, Params.WINDOW_SIZES, f_name, Params.WITH_TIME_DURATION)

    def test_kernels(self, new_file_name: str = "kernel_testing_ws"):
        features = self.dataset.get_features(Params.WINDOW_SIZE)
        SVM.test_kernels(features, self.__get_f_name(new_file_name, Params.WINDOW_SIZE))

    def test_kernel_c_gamma_parameters(self, new_file_name: str = "kernel_c_gamma_testing_ws"):
        features = self.dataset.get_features(Params.WINDOW_SIZE)
        SVM.test_kernel_c_gamma_parameters(features, Params.KERNEL, Params.C_REGULATIONS, Params.GAMMAS,
                                           self.__get_f_name(new_file_name, Params.WINDOW_SIZE),
                                           Params.WITH_TIME_DURATION)

    def test_pca(self, new_file_name: str = "pca_testing_ws"):
        features = self.dataset.get_features(Params.WINDOW_SIZE)
        SVM.test_pca(features, Params.PCA_N_COMPONENTS_LIST, self.__get_f_name(new_file_name, Params.WINDOW_SIZE))

    def test_best_svm(self):
        features = self.dataset.get_features(Params.WINDOW_SIZE)
        SVM.test_best_svm(features, self.dataset.get_activities(), Params.KERNEL, Params.C, Params.GAMMA)
