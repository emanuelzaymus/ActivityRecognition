from typing import Tuple

import numpy as np

from src.datasets.Dataset import Dataset
import src.data_file_handling as fh
import src.feature_extraction as fex


class Kyoto1(Dataset):
    __DIRECTORY = 'data/kyoto1'
    __ACTIVITIES = ['Phone_Call', 'Wash_hands', 'Cook', 'Eat', 'Clean']

    __FILES = ['p01', 'p02', 'p03', 'p04', 'p05', 'p06', 'p07', 'p08', 'p09', 'p10', 'p11', 'p12', 'p13', 'p14', 'p15',
               'p16', 'p32', 'p40', 'p41', 'p42', 'p43', 'p49', 'p50', 'p51']
    __EXTENSIONS = ['.t1', '.t2', '.t3', '.t4', '.t5']
    __EXTENSIONS_ACTIVITIES = [0, 1, 2, 3, 4]

    __SENSORS = ['AD1-A', 'AD1-B', 'AD1-C', 'D01', 'E01', 'I01', 'I02', 'I03', 'I04', 'I05', 'I06', 'I07', 'I08', 'M01',
                 'M07', 'M08', 'M09', 'M13', 'M14', 'M15', 'M16', 'M17', 'M18', 'M23', 'asterisk']

    @property
    def directory(self) -> str:
        return self.__DIRECTORY

    @property
    def activities(self) -> list:
        return self.__ACTIVITIES

    @property
    def files(self) -> list:
        return self.__FILES

    @property
    def extensions(self) -> list:
        return self.__EXTENSIONS

    @property
    def extensions_activities(self) -> list:
        return self.__EXTENSIONS_ACTIVITIES

    @property
    def sensors(self) -> list:
        return self.__SENSORS

    def get_features(self, windows_size: int, with_previous_class_feature: bool = False) -> np.ndarray:
        data_arrays, sensors = self.get_data_arrays()
        return fex.extract_features_from_arrays(data_arrays, windows_size, sensors, with_previous_class_feature)

    def get_activities(self) -> np.ndarray:
        return np.array(self.__ACTIVITIES)

    def get_data_arrays(self) -> Tuple[list, list]:
        return fh.get_data_arrays_from_directory(self), self.__SENSORS
