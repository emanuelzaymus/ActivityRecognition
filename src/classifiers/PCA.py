from typing import Tuple

import numpy as np
from sklearn.decomposition import PCA

import src.testing.TestParameters as Params


# noinspection PyPep8Naming
def decompose(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if Params.PCA:
        pca = PCA(Params.PCA_N_COMPONENTS)
        train_ = pca.fit_transform(X_train)
        test_ = pca.transform(X_test)
        # print("n components:", pca.n_components_)
        return train_, test_

    return X_train, X_test
