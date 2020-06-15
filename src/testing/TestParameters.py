from typing import List

from src.classifiers.Preprocessing import Preprocessing

""" 
TODO: comment for Parameters
"""

WITH_PREVIOUS_CLASS_FEATURE: bool = False

WINDOW_SIZES: List[int] = [5, 7, 10, 12, 15, 17, 19, 22, 25, 27, 30, 32, 35, 37, 40]  # [5, 12, 19, 30, 40]

PREPROCESSOR: Preprocessing = Preprocessing.STANDARD_SCALER

PCA: bool = True
PCA_N_COMPONENTS = None

RANDOM_STATE: int = 0
