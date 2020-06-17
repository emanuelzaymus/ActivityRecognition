from abc import abstractmethod
from typing import Tuple, List

import numpy as np

import src.data_file_handling as fh
import src.feature_extraction as fex
from src.datasets.Dataset import Dataset


class ArubaBase(Dataset):
    """ TODO Aruba comment """

    def get_features(self, windows_size: int, with_previous_class_feature: bool = False) -> np.ndarray:
        return fex.extract_features(self.__get_data_array(), windows_size,
                                    with_previous_class_feature=with_previous_class_feature)[0]

    def get_activities(self) -> np.ndarray:
        return np.array(self._activities)

    def get_data_arrays(self) -> Tuple[List[np.ndarray], List[str]]:
        data_array: np.ndarray = self.__get_data_array()
        fex.fill_missing_activities(data_array)
        return [data_array], self._sensors

    def __get_data_array(self) -> np.ndarray:
        return fh.get_data_array(self._file)

    @property
    @abstractmethod
    def _file(self) -> str:
        return self.__FILE

    @property
    @abstractmethod
    def _activities(self) -> List[str]:
        return self.__ACTIVITIES

    @property
    @abstractmethod
    def _sensors(self) -> List[str]:
        return self.__SENSORS
