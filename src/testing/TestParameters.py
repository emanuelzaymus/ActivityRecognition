from typing import List

from src.classifiers.Kernel import Kernel
from src.classifiers.Preprocessing import Preprocessing

""" 
TODO: comment for Parameters
"""

RANDOM_STATE: int = 0

WITH_PREVIOUS_CLASS_FEATURE: bool = False

# WINDOW_SIZES: List[int] = [5, 7, 10, 12, 15, 17, 19, 22, 25, 27, 30, 32, 35, 37, 40, 50, 60, 70]  # [5, 12, 19, 30, 40]
WINDOW_SIZES: List[int] = [5, 12, 19, 30, 40]

WINDOW_SIZE: int = 40

PREPROCESSOR: Preprocessing = Preprocessing.STANDARD_SCALER
# PREPROCESSOR: Preprocessing = Preprocessing.ROBUST_SCALER

PCA: bool = True

PCA_N_COMPONENTS_LIST: List[float] = [0.9999999999, .99, .95, .9, .85, .8, .75, .7, .6, .5]

PCA_N_COMPONENTS: float = .9

# C_REGULATIONS: List[int] = [1, 2, 5, 7, 10, 13, 15, 17, 20, 30, 50, 75, 100, 150, 200]
C_REGULATIONS: List[int] = [1, 5, 10, 17, 50, 150]

# GAMMAS: List[float] = [.1, .15, .2, .25, .3, .35, .4, .45]
GAMMAS: List[float] = [.1, .2, .3]

KERNEL: Kernel = Kernel.RBF
C = 150
GAMMA = 0.1
