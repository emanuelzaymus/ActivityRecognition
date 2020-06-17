from datetime import datetime

import numpy as np
import pandas as pd

from src.DataArray import DataArray


class _RawFileColumns:
    NUMBER_OF_COLUMNS = 5

    DATE = 0
    TIME = 1
    SENSOR = 2
    VALUE = 3
    # ACTIVITY = 4


def get_data_array(file_name: str, delimiter: str = None) -> np.ndarray:
    """
        Loads and converts data from file.

        **Data in the file needs to be in following format separated by TAB:**
            - DATE - YY-mm-dd
            - TIME - HH:MM:SS.ffffff (milliseconds *.ffffff* are optional and are ignored by algorithm)
            - SENSOR - name of the sensor
            - VALUE - value of the sensor - is ignored
            - ACTIVITY - optional, in format: *activity_name* *begin/start/end*
        **Returned Numpy array in format: [[datetime.datetime SENSOR ACTIVITY] ... ]**
            - datetime.datetime - created from DATE and TIME
            - SENSOR - name of the sensor, unchanged
            - ACTIVITY - empty activities are replaced by ``DataArray.NO_ACTIVITY``

    :param file_name: File path
    :param delimiter: Delimiter of the file data
    :returns: Loaded and converted data from file (np.ndarray)
    """
    try:
        with open(file_name, 'rb') as f:
            data: pd.DataFrame = __read_table(f, delimiter, 'utf8')
    except:
        with open(file_name, 'rb') as f:
            data: pd.DataFrame = __read_table(f, delimiter, 'utf16')

    data: np.ndarray = data.fillna(DataArray.NO_ACTIVITY).values
    __convert_to_datetime(data)
    __process_sensors(data)
    return __delete_unnecessary_columns(data)


def __read_table(file: str, delimiter: str, encoding: str) -> pd.DataFrame:
    return pd.read_table(file, delimiter=delimiter, header=None,
                         names=range(_RawFileColumns.NUMBER_OF_COLUMNS),
                         index_col=False,
                         encoding=encoding)


def __convert_to_datetime(data: np.ndarray):
    for row in data:
        date: str = row[_RawFileColumns.DATE].strip()
        time: str = row[_RawFileColumns.TIME].strip()

        try:
            datetime_object: datetime = datetime.strptime(date + ' ' + time[:8], '%Y-%m-%d %H:%M:%S')
        except:
            try:
                datetime_object: datetime = datetime.strptime(date + ' ' + time[:8], '%Y-%m-%d %H.%M.%S')
            except:
                datetime_object: datetime = datetime.strptime(date + ' ' + time[:5], '%Y-%m-%d %H:%M')

        row[DataArray.DATETIME] = datetime_object


def __process_sensors(data: np.ndarray):
    data[:, _RawFileColumns.SENSOR] = [x.split()[0] for x in data[:, _RawFileColumns.SENSOR]]


def __delete_unnecessary_columns(data: np.ndarray) -> np.ndarray:
    return np.delete(data, [_RawFileColumns.TIME, _RawFileColumns.VALUE], 1)
