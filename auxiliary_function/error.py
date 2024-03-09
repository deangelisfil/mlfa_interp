import numpy as np
from math import sqrt


def mse(x):
    """
    Computes 1/M sum_i^M x_i**2
    """
    return (x**2).mean().item()


def rmse(x):
    """
    Computes sqrt(1/M sum_i^M x_i**2)
    """
    return sqrt(mse(x))


def maxe(x):
    """
    Computes max_i |x_i|
    """
    assert isinstance(x, np.ndarray)
    return np.abs(x).max()