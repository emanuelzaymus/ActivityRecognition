import numpy as np

from sklearn.preprocessing import StandardScaler, RobustScaler, MaxAbsScaler, MinMaxScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def test_default_SVC(features):
    X, y = __split_features(features)
    clf = SVC()
    return __test(X, y, clf)


def test_kernels(features):
    X, y = __split_features(features)

    kernels = ('linear', 'poly', 'rbf', 'sigmoid')
    file_to_save = np.empty((len(kernels), 2), dtype='object')

    for i in range(len(kernels)):
        print("Kernel:", kernels[i])
        file_to_save[i, 0] = kernels[i]
        clf = SVC(kernel=kernels[i])

        accuracy = __test(X, y, clf)
        print("Accuracy:", accuracy)
        file_to_save[i, 1] = accuracy

    np.savetxt("kernel_testing.txt", file_to_save, delimiter='\t', fmt="%s")


def test_c_gamma_default_parameters(features):
    c_regulations = (.25, .5, 1, 2, 5, 7, 10, 13, 15, 17, 20, 30, 50, 75, 100, 150, 200, 500, 1000)
    gammas = ('scale', 'auto')

    file_to_save = np.empty((len(c_regulations) * len(gammas), 3), dtype=object)
    __test_c_gamma_parameters(features, c_regulations, gammas, file_to_save)
    np.savetxt("c_gamma_default_testing.txt", file_to_save, delimiter='\t', fmt="%s")


def test_c_gamma_parameters(features):
    c_regulations = (1, 2, 5, 7, 10, 13, 15, 17, 20, 30, 50, 75, 100, 150, 200)
    gammas = (.1, .15, .2, .25, .3, .35, .4, .45)

    file_to_save = np.empty((len(c_regulations) * len(gammas), 3), dtype=object)
    __test_c_gamma_parameters(features, c_regulations, gammas, file_to_save)
    np.savetxt("c_gamma_testing.txt", file_to_save, delimiter='\t', fmt="%s")


def test_best_SVC(features, activities=None):
    X, y = __split_features(features)
    clf = SVC(kernel='rbf', C=20, gamma=0.2)
    # clf = SVC(kernel='rbf', C=13, gamma=0.3)
    print("TOTAL accuracy score:", __test(X, y, clf, activities=activities))


def __split_features(features):
    X = features[:, :-1]
    y = features[:, -1]
    return X, y


def __test(X, y, clf, random_state=0, activities=None):
    scores = np.array([])

    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        clf.fit(X_train, y_train)
        predict = clf.predict(X_test)

        print("Report:\n", classification_report(y_test, predict, target_names=activities))
        print("Confusion matrix:\n", confusion_matrix(y_test, predict), end="\n\n")
        acc_score = accuracy_score(y_test, predict)
        scores = np.append(scores, acc_score)

    return np.mean(scores) * 100


def __test_c_gamma_parameters(features, c_regulations, gammas, file_to_save):
    X, y = __split_features(features)

    i = 0
    for c in c_regulations:
        for g in gammas:
            print("C:", c, " Gamma:", g)
            file_to_save[i, 0] = c
            file_to_save[i, 1] = g

            clf = SVC(kernel='rbf', C=c, gamma=g)

            accuracy = __test(X, y, clf)
            print("Accuracy:", accuracy)
            file_to_save[i, 2] = accuracy
            i += 1
