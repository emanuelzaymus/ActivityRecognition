import os
from typing import Tuple

import numpy as np
import pandas as pd

from src import data_file_handling as dfh
from src.DataArray import DataArray
from src.data_file_handling import _RawFileColumns
from src.datasets.Arbua import Aruba
from src.datasets.Dataset import Dataset
from src.datasets.Kyoto2 import Kyoto2
from src.datasets.Kyoto3 import Kyoto3
import src.feature_extraction as fex
import src.data_file_handling as fh


def __get_features_activities(windows_size: int, with_previous_class_feature: bool = False) -> Tuple[
    np.ndarray, np.ndarray]:
    return fex.extract_features(__get_data_array(), windows_size,
                                with_previous_class_feature=with_previous_class_feature)


def __get_data_array() -> np.ndarray:
    return fh.get_data_array(Aruba().file)


# a: np.ndarray = __get_features_activities(1)
a: np.ndarray = __get_data_array()

print(np.unique(a[:, 1]))
for i in np.unique(a[:, 1]):
    print("'" + i + "'" + ", ", end="")
