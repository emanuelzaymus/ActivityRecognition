from enum import Enum
from typing import Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler, Normalizer


# noinspection PyPep8Naming
class Preprocessing(Enum):
    NOTHING = None

    STANDARD_SCALER = StandardScaler()
    ROBUST_SCALER = RobustScaler()
    MIN_MAX_SCALER = MinMaxScaler()
    MAX_ABS_SCALER = MaxAbsScaler()

    NORMALIZER_L1 = Normalizer('l1')
    NORMALIZER_L2 = Normalizer('l2')
    NORMALIZER_MAX = Normalizer('max')

    def preprocess(self, X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self is not Preprocessing.NOTHING:
            train_ = self.value.fit_transform(X_train)
            test_ = self.value.transform(X_test)
            return train_, test_

        return X_train, X_test

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        if self is not Preprocessing.NOTHING:
            return self.value.fit_transform(X_train)

        return X_train

    def transform(self, feat_vector: np.ndarray) -> np.ndarray:
        if self is not Preprocessing.NOTHING:
            return self.value.transform(feat_vector)

        return feat_vector
