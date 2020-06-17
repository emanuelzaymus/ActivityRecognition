import os
from typing import Tuple, List

import numpy as np

import src.data_file_handling as fh
import src.feature_extraction as fex
from src.DataArray import DataArray
from src.datasets.Dataset import Dataset


class Kyoto3(Dataset):
    """ Dataset **ADL Interweaved Activities** (3 Kyoto) from http://casas.wsu.edu/datasets/ """

    __DIRECTORY = 'data/kyoto3'
    __ACTIVITIES = ['Fill_medication_dispenser', 'Watch_DVD', 'Water_plants', 'Answer_the_phone',
                    'Prepare_birthday_card', 'Prepare_soup', 'Clean', 'Choose_outfit']

    __EXTENSIONS = ['.t1', '.t2', '.t3', '.t4', '.t5', '.t6', '.t7', '.t8']
    __TXT_EXTENSIONS = ['.t1.txt', '.t2.txt', '.t3.txt', '.t4.txt', '.t5.txt', '.t6.txt', '.t7.txt', '.t8.txt']
    __EXTENSIONS_ACTIVITIES = [1, 2, 3, 4, 5, 6, 7, 8]

    __SENSORS = ['AD1-B', 'AD1-C', 'D07', 'D08', 'D09', 'D10', 'D11', 'D12', 'E01', 'I01', 'I02', 'I03', 'I04', 'I05',
                 'I06', 'I07', 'I08', 'I09', 'M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10',
                 'M11', 'M12', 'M13', 'M14', 'M15', 'M16', 'M17', 'M18', 'M19', 'M21', 'M22', 'M23', 'M24', 'M25',
                 'M26', 'M51', 'P01', 'T01', 'T02']

    __ALL_FILES_BY_DIRECTORIES = [
        ['p04/p04.t1', 'p04/p04.t2', 'p04/p04.t3', 'p04/p04.t4', 'p04/p04.t5', 'p04/p04.t6', 'p04/p04.t7',
         'p04/p04.t8'],
        ['p13/p13.t1', 'p13/p13.t2', 'p13/p13.t3', 'p13/p13.t4', 'p13/p13.t5', 'p13/p13.t6', 'p13/p13.t7',
         'p13/p13.t8'],
        ['p14/p14.t1', 'p14/p14.t2', 'p14/p14.t3', 'p14/p14.t4', 'p14/p14.t5', 'p14/p14.t6', 'p14/p14.t7',
         'p14/p14.t8'],
        ['p15/p15.t1', 'p15/p15.t2', 'p15/p15.t3', 'p15/p15.t4', 'p15/p15.t5', 'p15/p15.t6', 'p15/p15.t7',
         'p15/p15.t8'],
        ['p17/p17.t1', 'p17/p17.t2', 'p17/p17.t3', 'p17/p17.t4', 'p17/p17.t5', 'p17/p17.t6', 'p17/p17.t7',
         'p17/p17.t8'],
        ['p18/p18.t1.txt', 'p18/p18.t2.txt', 'p18/p18.t3.txt', 'p18/p18.t4.txt', 'p18/p18.t5.txt', 'p18/p18.t6.txt',
         'p18/p18.t7.txt', 'p18/p18.t8.txt'],
        ['p19/p19.t1', 'p19/p19.t2', 'p19/p19.t3', 'p19/p19.t4', 'p19/p19.t5', 'p19/p19.t6', 'p19/p19.t7',
         'p19/p19.t8'],
        ['p20/p20.t1', 'p20/p20.t2', 'p20/p20.t3', 'p20/p20.t4', 'p20/p20.t5', 'p20/p20.t6', 'p20/p20.t7',
         'p20/p20.t8'],
        ['p22/p22.t1', 'p22/p22.t2', 'p22/p22.t3', 'p22/p22.t4', 'p22/p22.t5', 'p22/p22.t6', 'p22/p22.t7',
         'p22/p22.t8'],
        ['p23/p23.t1', 'p23/p23.t2', 'p23/p23.t3', 'p23/p23.t4', 'p23/p23.t5', 'p23/p23.t6', 'p23/p23.t7',
         'p23/p23.t8'],
        ['p24/p24.t1', 'p24/p24.t2', 'p24/p24.t3', 'p24/p24.t4', 'p24/p24.t5', 'p24/p24.t6', 'p24/p24.t7',
         'p24/p24.t8'],
        ['p25/p25.t1', 'p25/p25.t2', 'p25/p25.t3', 'p25/p25.t4', 'p25/p25.t5', 'p25/p25.t6', 'p25/p25.t7',
         'p25/p25.t8'],
        ['p26/p26.t1', 'p26/p26.t2', 'p26/p26.t3', 'p26/p26.t4', 'p26/p26.t5', 'p26/p26.t6', 'p26/p26.t7',
         'p26/p26.t8'],
        ['p27/p27.t2', 'p27/p27.t3.txt', 'p27/p27.t4.txt', 'p27/p27.t5.txt', 'p27/p27.t6.txt', 'p27/p27.t7.txt',
         'p27/p27.t8.txt'],
        ['p28/p28.t1.txt', 'p28/p28.t2', 'p28/p28.t3.txt', 'p28/p28.t4.txt', 'p28/p28.t5.txt', 'p28/p28.t6.txt',
         'p28/p28.t7.txt', 'p28/p28.t8.txt'],
        ['p29/p29.t1.txt', 'p29/p29.t2.txt', 'p29/p29.t3.txt', 'p29/p29.t4.txt', 'p29/p29.t5.txt', 'p29/p29.t6.txt',
         'p29/p29.t7.txt', 'p29/p29.t8.txt'],
        ['p30/p30.t1.txt', 'p30/p30.t2.t2', 'p30/p30.t3.txt', 'p30/p30.t4.txt', 'p30/p30.t5.txt', 'p30/p30.t6.txt',
         'p30/p30.t7.txt', 'p30/p30.t8.txt'],
        ['p31/p31.t1.txt', 'p31/p31.t2.txt', 'p31/p31.t3.txt', 'p31/p31.t4.txt', 'p31/p31.t5.txt', 'p31/p31.t6.txt',
         'p31/p31.t7.txt', 'p31/p31.t8.txt'],
        ['p32/p32.t1.txt', 'p32/p32.t2.txt', 'p32/p32.t3.txt', 'p32/p32.t4.txt', 'p32/p32.t5.txt', 'p32/p32.t6.txt',
         'p32/p32.t7.txt', 'p32/p32.t8.txt'],
        ['p33/p33.t1.txt', 'p33/p33.t2.txt', 'p33/p33.t3.txt', 'p33/p33.t4.txt', 'p33/p33.t5.txt', 'p33/p33.t6.txt',
         'p33/p33.t7.txt', 'p33/p33.t8.txt'],
        ['p34/p34.t1.txt', 'p34/p34.t2.txt', 'p34/p34.t3.txt', 'p34/p34.t4.txt', 'p34/p34.t5.txt', 'p34/p34.t6.txt',
         'p34/p34.t7.txt', 'p34/p34.t8.txt']
    ]

    def get_features(self, windows_size: int, with_previous_class_feature: bool = False) -> np.ndarray:
        data_arrays, sensors = self.get_data_arrays()
        return fex.extract_features_from_arrays(data_arrays, windows_size, sensors, with_previous_class_feature)

    def get_activities(self) -> np.ndarray:
        return np.array(self.__ACTIVITIES)

    def get_data_arrays(self) -> Tuple[List[np.ndarray], List[str]]:
        return self.__get_data_arrays_from_directory(), self.__SENSORS

    def __get_data_arrays_from_directory(self, delimiter: str = None) -> List[np.ndarray]:
        ret_data_list = []

        for file_list in self.__ALL_FILES_BY_DIRECTORIES:
            one_recording: np.ndarray = None
            for file in file_list:
                current_file_path = os.path.join(self.__DIRECTORY, file)

                data = fh.get_data_array(current_file_path, delimiter)
                data[:, DataArray.ACTIVITY] = self.__get_activity_from_extension(file)  # Replace activity

                one_recording = data if file == file_list[0] else np.append(one_recording, data, axis=0)

            ret_data_list.append(one_recording)

        return ret_data_list

    def __get_activity_from_extension(self, file_name: str) -> int:
        if file_name[-3:] in self.__EXTENSIONS:
            i = self.__EXTENSIONS.index(file_name[-3:])

        else:  # elif file_name[-7:] in self.__TXT_EXTENSIONS:
            i = self.__TXT_EXTENSIONS.index(file_name[-7:])

        return self.__EXTENSIONS_ACTIVITIES[i]
