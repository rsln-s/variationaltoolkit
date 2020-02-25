#!/usr/bin/env python

# To run: for i in {1..20}; do time python cobyla_opt.py; done

import numpy as np
import networkx as nx
import pickle
import argparse
from functools import partial
from variationaltoolkit import ObjectiveWrapper
from qiskit.optimization.ising.max_cut import get_operator as get_maxcut_operator
import scipy

def maxcut_obj(x,G):
    cut = 0
    for i, j in G.edges():
        if x[i] != x[j]:
            # the edge is cut
            cut -= 1
    return cut


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--maxiter", type = int,
        default = "300",
        help = "number of iterations, default is 100")
    parser.add_argument(
        "-p", type = int,
        default = "10",
        help = "maximum depth to explore")
    args = parser.parse_args()

    #import logging; logging.basicConfig(level=logging.INFO)

    # For testing purposes, hardcode Peterson graph

    elist = [
        [0,1],[1,2],[2,3],[3,4],[4,0],
        [0,5],[1,6],[2,7],[3,8],[4,9],
        [5,7],[5,8],[6,8],[6,9],[7,9]
    ]
    
    G=nx.OrderedGraph()
    G.add_edges_from(elist)

    w = nx.adjacency_matrix(G)
    obj_f_cut = partial(maxcut_obj, G=G)
    C, _ = get_maxcut_operator(w)

    lb = np.array([0, 0] * args.p)
    ub = np.array([np.pi / 2] * args.p + [np.pi] * args.p)

    np.random.seed(0)
    init_theta = np.random.uniform(lb, ub)

    obj_w = ObjectiveWrapper(
            obj_f_cut, 
            varform_description={'name':'QAOA', 'p':args.p, 'cost_operator':C, 'num_qubits':G.number_of_nodes()}, 
            backend_description={'package':'qiskit', 'provider':'Aer', 'name':'statevector_simulator'},
            execute_parameters={})

    assert(obj_w.num_parameters == 2*args.p)

    res = scipy.optimize.minimize(obj_w.get_obj(), init_theta, method='COBYLA', options={'maxiter':args.maxiter})
    print(res)
