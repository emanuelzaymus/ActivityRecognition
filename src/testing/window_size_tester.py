import numpy as np

from src.classifiers import SVM
import src.feature_extraction as fex


def test_variable_window_sizes(data_arrays: list, sensors: list, window_sizes: list = None,
                               with_previous_class_feature: bool = False, fname_to_save: str = None):
    if window_sizes is None:
        window_sizes = [5, 7, 10, 13, 15, 17, 20, 23, 25, 27, 30, 33, 35, 37, 38, 39, 40]

    file_to_save: np.ndarray = np.zeros((len(window_sizes), 2))

    for i in range(len(window_sizes)):
        print("Window size:", window_sizes[i])
        file_to_save[i, 0] = window_sizes[i]
        features = fex.extract_features_from_arrays(data_arrays, window_sizes[i], sensors,
                                                    with_previous_class_feature=with_previous_class_feature)

        accuracy: float = SVM.test_default_SVC(features, with_previous_class_feature=with_previous_class_feature)
        print("Accuracy:", accuracy, '\n')
        file_to_save[i, 1] = accuracy

    if fname_to_save is not None:
        np.savetxt("results/" + fname_to_save, file_to_save, delimiter='\t', fmt="%.4f")


def test_variable_window_sizes_aruba(data_array: list, window_sizes: list = None,
                                     with_previous_class_feature: bool = False, fname_to_save: str = None):
    if window_sizes is None:
        window_sizes = [5, 7, 10, 13, 15, 17, 20, 23, 25, 27, 30, 33, 35, 37, 38, 39, 40]

    file_to_save: np.ndarray = np.zeros((len(window_sizes), 2))

    for i in range(len(window_sizes)):
        print("Window size:", window_sizes[i])
        file_to_save[i, 0] = window_sizes[i]
        features, a = fex.extract_features(data_array, window_sizes[i],
                                           with_previous_class_feature=with_previous_class_feature)

        accuracy: float = SVM.test_default_SVC(features=features,
                                               with_previous_class_feature=with_previous_class_feature)
        print("Accuracy:", accuracy, '\n')
        file_to_save[i, 1] = accuracy

    if fname_to_save is not None:
        np.savetxt("results/" + fname_to_save, file_to_save, delimiter='\t', fmt="%.4f")
