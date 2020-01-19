import numpy as np
import networkx as nx

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


def modularity_obj(x, N, G, node_list):
    """Compute -1 times the value of modularity.
    Args:
        x (numpy.ndarray): binary string as numpy array.
        N: number of vars per node
        G: graph(each node needs to have attribute 'volume', each edge needs to have attribute 'weight')
        node_list: list of the nodes in the graph
    Returns:
        float: value of the modularity.
    """
    ptn_variables = {}
    n = G.number_of_nodes()
    node_dict = {node_list[i]: i for i in range(n)}
    if 'Cluster0' in G:
        for node in node_list[:n-2**N]:
            ptn = bin_to_dec(x[node_dict[node]*N: (node_dict[node]+1)*N], N)
            ptn_variables[node] = ptn
        for i in range(2**N):
            ptn_variables['Cluster'+str(i)] = i
    else:
        for node in node_list:
            ptn = bin_to_dec(x[node_dict[node]*N: (node_dict[node]+1)*N], N)
            ptn_variables[node] = ptn

    obj = compute_objective(G, ptn_variables, N)
    
    return -obj


def compute_objective(G, ptn_variables, N):
# helper function to compute modularity_obj

    parts = 2**N
    ptn_properties = {i: {'sum_of_vols':0.0, 'edge_cut': 0.0} for i in range(parts)}
    ptn_properties['total_edges'] = 0

    for node in G.nodes():
        part = ptn_variables[node]
        ptn_properties[part]['sum_of_vols'] += G.nodes[node]['volume']
        ptn_properties['total_edges'] += G.nodes[node]['volume']

    for u, v in G.edges():
        partu = ptn_variables[u]
        partv = ptn_variables[v]
        if partu != partv:
            ptn_properties[partu]['edge_cut'] += G[u][v]['weight']
            ptn_properties[partv]['edge_cut'] += G[u][v]['weight']

    obj = 0
    for part in range(parts):
        obj += (ptn_properties[part]['sum_of_vols'] - ptn_properties[part]['edge_cut']) / ptn_properties['total_edges']
        obj -= (ptn_properties[part]['sum_of_vols'] / ptn_properties['total_edges']) ** 2

    return obj

def bin_to_dec(x, N):
# helper function to compute modularity_obj
    sum = 0
    for j in range(N):
        sum += 2**j * int(x[j])
    
    return sum