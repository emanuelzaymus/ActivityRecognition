import os
from typing import Tuple

import numpy as np
import pandas as pd

from src import data_file_handling as dfh
from src.DataArray import DataArray
from src.classifiers.Kernel import Kernel
from src.data_file_handling import _RawFileColumns
from src.datasets.Aruba import Aruba
from src.datasets.Dataset import Dataset
from src.datasets.Kyoto2 import Kyoto2
from src.datasets.Kyoto3 import Kyoto3
import src.feature_extraction as fex
import src.data_file_handling as fh

a: np.ndarray = fh.get_data_array("data/data_aruba_formatted_2months.txt")

print(a.shape)

b = np.unique(a[:, 2])

print(b)
