import os
from datetime import datetime

import numpy as np
import pandas as pd

from src.DataArray import DataArray
from src.datasets.Dataset import Dataset


class _RawFileColumns:
    NUMBER_OF_COLUMNS = 5

    DATE = 0
    TIME = 1
    # SENSOR = 2
    VALUE = 3
    # ACTIVITY = 4


def get_data_arrays_from_directory(dataset: Dataset, delimiter: str = None) -> list:
    ret_data_list = []

    for file in dataset.files:
        one_recording: np.ndarray = None
        for i in range(len(dataset.extensions_activities)):
            current_file_path = os.path.join(dataset.directory, file + dataset.extensions[i])

            data = get_data_array(current_file_path, delimiter)
            data[:, DataArray.ACTIVITY] = dataset.extensions_activities[i]

            one_recording = data if i == 0 else np.append(one_recording, data, axis=0)

        ret_data_list.append(one_recording)

    return ret_data_list


def get_data_array(file_name: str, delimiter: str = None) -> np.ndarray:
    """
        Loads and converts data from file.

        **Data in the file needs to be in following format separated by TAB:**
            - DATE - YY-mm-dd
            - TIME - HH:MM:SS.ffffff (milliseconds *.ffffff* are optional and are ignored by algorithm)
            - SENSOR - name of the sensor
            - VALUE - value of the sensor - is ignored
            - ACTIVITY - optional, in format: *activity_name* *begin/start/end*
        **Returned Numpy array in format: [[datetime.datetime SENSOR ACTIVITY]...]**
            - datetime.datetime - created from DATE and TIME
            - SENSOR - name of the sensor, unchanged
            - ACTIVITY - empty activities are replaced by ``DataArray.NO_ACTIVITY``
    Parameters:
        file_name (string): File path
        delimiter (string): Delimiter of the file data
    Returns:
        ndarray: Loaded and converted data from file
    """
    data: pd.DataFrame = pd.read_table(file_name, delimiter=delimiter, header=None,
                                       names=range(_RawFileColumns.NUMBER_OF_COLUMNS),
                                       index_col=False)
    data: np.ndarray = data.fillna(DataArray.NO_ACTIVITY).values
    __convert_to_datetime(data)
    return __delete_unnecessary_columns(data)


def __convert_to_datetime(data: np.ndarray):
    for row in data:
        date: str = row[_RawFileColumns.DATE].strip()
        time: str = row[_RawFileColumns.TIME].strip()

        datetime_object: datetime = datetime.strptime(date + ' ' + time[:8], '%Y-%m-%d %H:%M:%S')
        row[DataArray.DATETIME] = datetime_object


def __delete_unnecessary_columns(data: np.ndarray) -> np.ndarray:
    return np.delete(data, [_RawFileColumns.TIME, _RawFileColumns.VALUE], 1)
