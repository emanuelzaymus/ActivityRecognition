from abc import abstractmethod
from typing import Tuple
import numpy as np


class Dataset:

    @property
    @abstractmethod
    def directory(self) -> str:
        pass

    @property
    @abstractmethod
    def activities(self) -> list:
        pass

    @property
    @abstractmethod
    def files(self) -> list:
        pass

    @property
    @abstractmethod
    def extensions(self) -> list:
        pass

    @property
    @abstractmethod
    def extensions_activities(self) -> list:
        pass

    @property
    @abstractmethod
    def sensors(self) -> list:
        pass

    @abstractmethod
    def get_features(self, windows_size: int, with_previous_class_feature: bool = False) -> np.ndarray:
        pass

    @abstractmethod
    def get_activities(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_data_arrays(self) -> Tuple[list, list]:
        pass
