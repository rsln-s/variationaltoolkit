import numpy as np

"""

Specifies all kinds of objectives

"""

def maxcut_obj(x,w):
    """Compute the value of a cut.
    Args:
        x (numpy.ndarray): binary string as numpy array.
        w (numpy.ndarray): adjacency matrix.
    Returns:
        float: value of the cut.
    """
    X = np.outer(x, (1 - x))
    return np.sum(w * X)

