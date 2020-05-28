import numpy as np

from src import data_file_handling as dfh

a: np.ndarray = dfh.get_data_array('data/data_aruba_formatted_5days.txt')
b = np.unique(a[:, 2])
for i in b:
    print(i)
