from typing import List

from src.datasets.aruba.ArubaBase import ArubaBase


class Aruba2Months(ArubaBase):
    """ TODO Aruba 2 months comment """

    __FILE = 'data/data_aruba_formatted_2months.txt'

    __ACTIVITIES = ['Bed_to_Toilet', 'Eating', 'Enter_Home', 'Housekeeping', 'Leave_Home', 'Meal_Preparation', 'Relax',
                    'Respirate', 'Sleeping', 'Wash_Dishes', 'Work', 'Nothing']

    __SENSORS = ['D001', 'D002', 'D004', 'M001', 'M002', 'M003', 'M004', 'M005', 'M006', 'M007', 'M008', 'M009', 'M010',
                 'M011', 'M012', 'M013', 'M014', 'M015', 'M016', 'M017', 'M018', 'M019', 'M020', 'M021', 'M022', 'M023',
                 'M024', 'M025', 'M026', 'M027', 'M028', 'M029', 'M030', 'M031', 'T001', 'T002', 'T003', 'T004', 'T005']

    @property
    def _file(self) -> str:
        return self.__FILE

    @property
    def _activities(self):
        return self.__ACTIVITIES

    @property
    def _sensors(self) -> List[str]:
        return self.__SENSORS
