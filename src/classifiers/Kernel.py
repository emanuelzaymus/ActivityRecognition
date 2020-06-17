from enum import Enum


class Kernel(Enum):
    LINEAR = 'linear'
    POLYNOMIAL = 'poly'
    RBF = 'rbf'
    SIGMOID = 'sigmoid'
