import numpy as np

"""

Specifies all kinds of objectives

"""

def maxcut_obj(x,w):
    """Compute -1 times the value of a cut.
    Args:
        x (numpy.ndarray): binary string as numpy array.
        w (numpy.ndarray): adjacency matrix.
    Returns:
        float: value of the cut.
    """
    X = np.outer(x, (1 - x))
    return -np.sum(w * X)


def modularity_obj(x, N, B, m):
    """Compute -1 times the value of modularity.
    Args:
        x (numpy.ndarray): binary string as numpy array.
        N (int):           number of variables per node.
        B (numpy.ndarray): modularity matrix.
        m (int):           number of edges.
    Returns:
        float: -1 times the value of the modularity.
    """
    obj = 0
    n = len(B)
    y = {}
    
    for i in range(n):
        y[i] = bin_to_dec(x[N*i: N*(i+1)], N)
    for i in range(n):
        for j in range(n):
            obj += B.item(i, j) * (y[i] == y[j])
            
    return -obj/2/m


def bin_to_dec(x, N):
# helper function to compute modularity_obj
    sum = 0
    for j in range(N):
        sum += 2**j * int(x[j])
    
    return sum
