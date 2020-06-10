import os
from abc import abstractmethod
from typing import Tuple, List

import numpy as np

import src.data_file_handling as fh
import src.feature_extraction as fex
from src.DataArray import DataArray
from src.datasets.Dataset import Dataset


class KyotoBase(Dataset):

    def get_features(self, windows_size: int, with_previous_class_feature: bool = False) -> np.ndarray:
        data_arrays, sensors = self.get_data_arrays()
        return fex.extract_features_from_arrays(data_arrays, windows_size, sensors, with_previous_class_feature)

    def get_activities(self) -> np.ndarray:
        return np.array(self._activities)

    def get_data_arrays(self) -> Tuple[List[np.ndarray], List[str]]:
        return self.__get_data_arrays_from_directory(), self._sensors

    def __get_data_arrays_from_directory(self, delimiter: str = None) -> List[np.ndarray]:
        ret_data_list = []

        for file in self._files:
            one_recording: np.ndarray = None
            for i in range(len(self._extensions_activities)):
                current_file_path = os.path.join(self._directory, file + self._extensions[i])

                data = fh.get_data_array(current_file_path, delimiter)
                data[:, DataArray.ACTIVITY] = self._extensions_activities[i]  # Replace activity

                one_recording = data if i == 0 else np.append(one_recording, data, axis=0)

            ret_data_list.append(one_recording)

        return ret_data_list

    @property
    @abstractmethod
    def _directory(self) -> str:
        pass

    @property
    @abstractmethod
    def _activities(self) -> List[str]:
        pass

    @property
    @abstractmethod
    def _files(self) -> List[str]:
        pass

    @property
    @abstractmethod
    def _extensions(self) -> List[str]:
        pass

    @property
    @abstractmethod
    def _extensions_activities(self) -> List[int]:
        pass

    @property
    @abstractmethod
    def _sensors(self) -> List[str]:
        pass
