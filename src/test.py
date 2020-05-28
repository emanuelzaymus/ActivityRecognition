import os

import numpy as np
import pandas as pd

from src import data_file_handling as dfh
from src.DataArray import DataArray
from src.data_file_handling import _RawFileColumns
from src.datasets.Dataset import Dataset
from src.datasets.Kyoto2 import Kyoto2

ret_data_list: np.ndarray = None
dataset: Dataset = Kyoto2()
for file in dataset.files:
    one_recording: np.ndarray = None
    for i in range(len(dataset.extensions_activities)):
        current_file_path = os.path.join(dataset.directory, file + dataset.extensions[i])

        data: pd.DataFrame = pd.read_table(current_file_path, delimiter=None, header=None,
                                           names=range(_RawFileColumns.NUMBER_OF_COLUMNS),
                                           index_col=False)

        # data[:, DataArray.ACTIVITY] = dataset.extensions_activities[i]  # Replace activity

        one_recording = data if i == 0 else np.append(one_recording, data, axis=0)

    # ret_data_list.append(one_recording)
    ret_data_list = one_recording if file == dataset.files[0] else np.append(ret_data_list, one_recording, axis=0)

print(ret_data_list)
print(ret_data_list.shape)

print(np.unique(ret_data_list[:, 2]))
