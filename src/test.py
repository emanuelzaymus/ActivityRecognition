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

# with open("data/kyoto3/p29/p29.t8.txt", 'rb') as f:
# with open("data/kyoto3/p30/p30.t1.txt", 'rb') as f:
#     data: pd.DataFrame = pd.read_table(f, header=None,
#                                        names=range(_RawFileColumns.NUMBER_OF_COLUMNS),
#                                        index_col=False,
#                                        encoding='utf16')
#
#     print(data)

a = np.array([["asdf pop", 8], ["asdf pop", 8], ["asdf sdsdscs  ", 8], [")SD PLP", 54654], ["wer POPOPO", 9]])

print(a)
print(np.shape(a[:, 0]))

# for i in range(np.shape(a[:, 0])[0]):
#     a[i, 0] = a[i, 0].split()[0]

a[:, 0] = [x.split()[0] for x in a[:, 0]]
# a[i, 0] = a[i, 0].split()[0]


print(a[:, 0])
print(a)
