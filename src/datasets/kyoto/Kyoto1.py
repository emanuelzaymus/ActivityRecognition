from typing import List

from src.datasets.kyoto.KyotoBase import KyotoBase


class Kyoto1(KyotoBase):
    """ Dataset **ADL Normal Activities** (1 Kyoto) from http://casas.wsu.edu/datasets/ """

    __DIRECTORY = 'data/kyoto1'
    __ACTIVITIES = ['Phone_Call', 'Wash_hands', 'Cook', 'Eat', 'Clean']

    __FILES = ['p01', 'p02', 'p03', 'p04', 'p05', 'p06', 'p07', 'p08', 'p09', 'p10', 'p11', 'p12', 'p13', 'p14', 'p15',
               'p16', 'p32', 'p40', 'p41', 'p42', 'p43', 'p49', 'p50', 'p51']
    __EXTENSIONS = ['.t1', '.t2', '.t3', '.t4', '.t5']
    __EXTENSIONS_ACTIVITIES = [0, 1, 2, 3, 4]

    __SENSORS = ['AD1-A', 'AD1-B', 'AD1-C', 'D01', 'E01', 'I01', 'I02', 'I03', 'I04', 'I05', 'I06', 'I07', 'I08', 'M01',
                 'M07', 'M08', 'M09', 'M13', 'M14', 'M15', 'M16', 'M17', 'M18', 'M23', 'asterisk']

    @property
    def _directory(self) -> str:
        return self.__DIRECTORY

    @property
    def _activities(self) -> List[str]:
        return self.__ACTIVITIES

    @property
    def _files(self) -> List[str]:
        return self.__FILES

    @property
    def _extensions(self) -> List[str]:
        return self.__EXTENSIONS

    @property
    def _extensions_activities(self) -> List[int]:
        return self.__EXTENSIONS_ACTIVITIES

    @property
    def _sensors(self) -> List[str]:
        return self.__SENSORS
