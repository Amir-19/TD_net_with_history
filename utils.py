import numpy as np

def sigmoid(x):
    try:
        res = 1 / (1 + np.exp(-x))
    except OverflowError:
        res = 0.0
    return res