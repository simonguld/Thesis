# Author: Simon Guldager Andersen
# Date(last edit): Nov. 11 - 2024

## Imports:
import os
import sys
import warnings
import time
import shutil
import io
import lz4
import json
import pickle


import sys
import os
import pickle as pkl
import warnings
import time

from functools import wraps, partial
from pathlib import Path
from multiprocessing import cpu_count
from multiprocessing.pool import Pool as Pool

import numpy as np
import matplotlib.pyplot as plt


import massPy as mp

sys.path.append('C:\\Users\\Simon Andersen\\Projects\\Projects\\Appstat2022\\External_Functions')
from ExternalFunctions import Chi2Regression


# Helper functions -------------------------------------------------------------------

def timeit(func):
    @wraps(func)   # keeps function name/docs intact
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)   # call the real function
        end = time.perf_counter()
        print(f"{func.__name__} runtime: {end - start:.3f} s")
        return result
    return wrapper


# CID related functions -------------------------------------------------------------------

def get_allowed_time_intervals(system_size, nbits_max = 8):
    """
    Get allowed intervals for CID calculation based on system size and max bits.
    """
    allowed_intervals = []
    
    # system size must be divisible by 2^n

    if np.log2(system_size) % 1 != 0:
        warnings.warn("System size must be a power of 2 for exact interval calculation.")
        return
    if not type(nbits_max) == int:
        raise ValueError("nbits_max must be an integer.")

    for nbits in range(1, nbits_max + 1):
        interval_exp = 3 * nbits - 2 * np.log2(system_size)

        if interval_exp < 0:
            continue

        allowed_intervals.append({'time_interval': int(2 ** interval_exp), 'nbits': nbits})
    return allowed_intervals


def block_flatten(array, m, k):
    """
    Efficiently flatten a 2D array into m x k blocks traversed horizontally.
    
    Parameters:
        array (np.ndarray): Input 2D array of shape (M, N)
        m (int): Number of rows per block
        k (int): Number of columns per block
        
    Returns:
        np.ndarray: Flattened 1D array of blocks
    """
    M, N = array.shape

    # Check divisibility
    if M % m != 0:
        raise ValueError(f"Number of rows {M} is not divisible by block row size {m}.")
    if N % k != 0:
        raise ValueError(f"Number of columns {N} is not divisible by block column size {k}.")
    
    # Reshape array into blocks
    reshaped = array.reshape(M//m, m, N//k, k)
    # Transpose to bring blocks into row-major order: (block_row, block_col, m, k)
    reshaped = reshaped.transpose(0, 2, 1, 3)
    # Flatten all blocks
    return reshaped.reshape(-1)
