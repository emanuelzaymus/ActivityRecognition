import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler, MinMaxScaler
import data_file_handling as fh
import feature_extraction as fex
import SVM


def test_variable_window_sizes(data, window_sizes=(5, 7, 10, 13, 15, 17, 20, 23, 25, 27, 30, 33, 35, 37, 40)):
    file_to_save = np.zeros((len(window_sizes), 2))

    for i in range(len(window_sizes)):
        print("Window size:", window_sizes[i])
        file_to_save[i, 0] = window_sizes[i]
        features, activities = fex.extract_features(data, window_sizes[i])

        accuracy = SVM.test_default_SVC(features)
        print("Accuracy:", accuracy, '\n')
        file_to_save[i, 1] = accuracy

    np.savetxt("window_size_testing.txt", file_to_save, delimiter='\t', fmt="%.4f")


def plot(feature_array, activities):
    X, y_ = SVM.__split_features(feature_array)

    sc = StandardScaler()
    # sc = MaxAbsScaler()
    # sc = MinMaxScaler()
    X = sc.fit_transform(X)

    X_items = np.column_stack((X[:, :13], X[:, -1]))  # item sensors
    X_motion = X[:, 13:-1]  # 13..-1 => motion sensors

    """ Booling the data """
    # for x in range(X_items.shape[0]):
    #     for y in range(X_items.shape[1]):
    #         X_items[x, y] = 1 if X_items[x, y] >= 1 else 0
    # for x in range(X_motion.shape[0]):
    #     for y in range(X_motion.shape[1]):
    #         X_motion[x, y] = 1 if X_motion[x, y] >= 1 else 0

    X_items = X_items.sum(axis=1)
    X_motion = X_motion.sum(axis=1)

    """ Print all the data """
    # for x in range(X_items.shape[0]):
    #     print("%3d" % (X_items[x]), end=" ")
    # print()
    # for x in range(X_motion.shape[0]):
    #     print("%3d" % (X_motion[x]), end=" ")
    # print()

    fig, ax = plt.subplots()

    scatter = ax.scatter(X_items, X_motion, c=y_, s=30, alpha=0.4, edgecolors=None)
    plt.title("Standard-Scaler Transforming")
    # plt.title("Min-Max-Scaler Transforming")
    # plt.title("Max-Abs-Scaler Transforming")
    plt.xlabel("Count of all ITEM sensors")
    plt.ylabel("Count of all MOTION sensors")
    plt.legend(*scatter.legend_elements(), loc="upper right", title="Activities")
    plt.show()


# data = fh.get_data_array("data_kyoto_1.txt")
# test_variable_window_sizes(data)

# features, activities = fex.extract_features(data, window_size=30)
# np.savetxt("adlnormal_features_ws30.txt", features, delimiter='\t', fmt="%d")
# np.savetxt("activities.txt", activities, delimiter='\t', fmt="%s")

features = pd.read_table("adlnormal_features_ws30.txt", header=None).values
activities = pd.read_csv("activities.txt", delimiter='\t', header=None).values.flatten()

# SVM.test_kernels(features)

# SVM.test_c_gamma_default_parameters(features)

# SVM.test_c_gamma_parameters(features)

SVM.test_best_SVC(features, activities)

# plot(features, activities)
