import numpy as np
from sklearn.metrics import accuracy_score

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# X = np.arange(10 * 3).reshape((10, 3))
#
# print(X)
#
# kf = KFold(n_splits=5, shuffle=False)
# for train_index, test_index in kf.split(X):
#     # print(train_index)
#     # print(test_index)
#
#     X_train, X_test = X[train_index], X[test_index]
#     # y_train, y_test = y[train_index], y[test_index]
#     sc = StandardScaler()
#
#     sc.fit(X_train)
#     X_train = sc.fit_transform(X_train)
#
#     print(X_test)
#     vector: np.ndarray
#     for vector in X_test:
#         print(vector)
#         vector = vector.reshape(1, -1)
#         print(vector)
#
#         vector_scaled = sc.fit_transform(vector)
#
#         print(vector_scaled)
#
#     exit(88)

# y_train = np.array([6,8,7,6,5,4,3,2,1,5,2,6,8,5,5,7,4,9,5,5,3,6,2,1,5,2,3,6,8,5,5,5,2,4,5,5,6,9,85,3,5,6,59,8,5,85,7,5,4,5,45,4,5,15,2,5,25,5])
# counts = np.bincount(y_train)
# last_class = np.argmax(counts)
# is_first = False
# print(counts)
# print(last_class)
# print(y_train[-1])
#
# if not 8 > 5:
#     sss = 654223
#
# print(sss)
