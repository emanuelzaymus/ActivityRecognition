from typing import List

from src.classifiers.Kernel import Kernel
from src.classifiers.Preprocessing import Preprocessing

""" 
TODO: comment for Parameters
"""

# Randomize
RANDOM_STATE: int = 0

# Window size
WINDOW_SIZES: List[int] = [5, 7, 10, 12, 15, 17, 19, 22, 25, 27, 30, 32, 35, 37, 40, 50, 60, 70]  # [5, 12, 19, 30, 40]
WINDOW_SIZE: int = 40

# Preprocessing and PCA
PREPROCESSOR: Preprocessing = Preprocessing.STANDARD_SCALER

PCA: bool = True
PCA_N_COMPONENTS_LIST: List[float] = [0.9999999999, .99, .95, .9, .85, .8, .75, .7, .6, .5]
PCA_N_COMPONENTS: float = .9

# SVMs parameters
KERNEL: Kernel = Kernel.RBF

C_REGULATIONS: List[int] = [1, 2, 5, 7, 10, 13, 15, 17, 20, 30, 50, 75, 100, 150, 200]  # [1, 5, 10, 17, 50, 150]
GAMMAS: List[float] = [.1, .15, .2, .25, .3, .35, .4, .45]  # [.1, .2, .3]

C = 150
GAMMA = 0.1

# Test with
WITH_TIME_DURATION = False
