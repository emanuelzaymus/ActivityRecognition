from typing import Tuple, List

import numpy as np

import src.data_file_handling as fh
import src.feature_extraction as fex
from src.datasets.Dataset import Dataset


class Aruba(Dataset):
    """ TODO Aruba comment """
    __FILE = 'data/data_aruba_formatted_5days.txt'

    __ACTIVITIES = ['Bed_to_Toilet', 'Eating', 'Enter_Home', 'Housekeeping', 'Leave_Home', 'Meal_Preparation', 'Relax',
                    'Sleeping', 'Wash_Dishes', 'Work', 'Nothing']

    __SENSORS = ['D001', 'D002', 'D004', 'M001', 'M002', 'M003', 'M004', 'M005', 'M006', 'M007', 'M008', 'M009', 'M010',
                 'M011', 'M012', 'M013', 'M014', 'M015', 'M016', 'M017', 'M018', 'M019', 'M020', 'M021', 'M022', 'M023',
                 'M024', 'M025', 'M026', 'M027', 'M028', 'M029', 'M030', 'M031', 'T001', 'T002', 'T003', 'T004', 'T005']

    def get_features(self, windows_size: int, with_previous_class_feature: bool = False) -> np.ndarray:
        return fex.extract_features(self.__get_data_array(), windows_size,
                                    with_previous_class_feature=with_previous_class_feature)[0]

    def get_activities(self) -> np.ndarray:
        return np.array(self.__ACTIVITIES)

    def get_data_arrays(self) -> Tuple[List[np.ndarray], List[str]]:
        data_array: np.ndarray = self.__get_data_array()
        fex.fill_missing_activities(data_array)
        return [data_array], self.__SENSORS

    def __get_data_array(self) -> np.ndarray:
        return fh.get_data_array(self.__FILE)
