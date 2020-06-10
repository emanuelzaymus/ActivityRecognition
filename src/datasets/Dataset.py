from abc import abstractmethod
from typing import Tuple

import numpy as np


class Dataset:

    @abstractmethod
    def get_features(self, windows_size: int, with_previous_class_feature: bool = False) -> np.ndarray:
        pass

    @abstractmethod
    def get_activities(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_data_arrays(self) -> Tuple[list, list]:
        pass
