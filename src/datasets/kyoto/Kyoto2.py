from typing import List

from src.datasets.kyoto.KyotoBase import KyotoBase


class Kyoto2(KyotoBase):
    """ Dataset **ADL Activities with Errors** (2 Kyoto) from http://casas.wsu.edu/datasets/ """

    __DIRECTORY = 'data/kyoto2'
    __ACTIVITIES = ['Phone_Call', 'Wash_hands', 'Cook', 'Eat', 'Clean']

    __FILES = ['p17', 'p18', 'p20', 'p21', 'p22', 'p23', 'p24', 'p26', 'p27', 'p29', 'p30', 'p31', 'p52', 'p53', 'p54',
               'p55', 'p56', 'p57', 'p58', 'p59']
    __EXTENSIONS = ['.t1', '.t2', '.t3', '.t4', '.t5']
    __EXTENSIONS_ACTIVITIES = [0, 1, 2, 3, 4]

    __SENSORS = ['AD1-A', 'AD1-B', 'AD1-C', 'D01', 'E01', 'I01', 'I02', 'I03', 'I04', 'I05', 'I06', 'I07', 'I08', 'I09',
                 'M01', 'M06', 'M07', 'M08', 'M09', 'M10', 'M13', 'M14', 'M15', 'M16', 'M17', 'M18', 'M21', 'M22',
                 'M23', 'asterisk']

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
