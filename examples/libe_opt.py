#!/usr/bin/env python
# Libensemble for parameter optimization

from __future__ import division
from __future__ import absolute_import

from mpi4py import MPI  # for libE communicator
import sys, os  # for adding to path
import numpy as np
import socket

from libensemble.libE import libE
from libensemble.gen_funcs.persistent_aposmm import aposmm as gen_f
from libensemble.alloc_funcs.persistent_aposmm_alloc import persistent_aposmm_alloc as alloc_f

import numpy as np
import networkx as nx
import pickle
import argparse
from functools import partial
from variationaltoolkit import ObjectiveWrapper
from qiskit.optimization.ising.max_cut import get_operator as get_maxcut_operator
from libensemble.utils import parse_args
nworkers, _, _, _ = parse_args()

def optimize_obj(obj_val, num_parameters, ub=None, lb=None, sim_max=None):

    def sim_func(H, gen_info, sim_specs, libE_info):
        del libE_info  # Ignored parameter

        batch = len(H['x'])
        O = np.zeros(batch, dtype=sim_specs['out'])

        for i, x in enumerate(H['x']):
            O['f'][i] = obj_val(x)

        print(O, flush=True)
        return O, gen_info

    script_name = os.path.splitext(os.path.basename(__file__))[0]
    #State the objective function, its arguments, output, and necessary parameters (and their sizes)
    sim_specs = {
        'sim_f':
            sim_func,  # This is the function whose output is being minimized
        'in': ['x'],  # These keys will be given to the above function
        'out': [
            ('f',
             float),  # This is the output from the function being minimized
        ],
    }
    gen_out = [
        ('x', float, num_parameters),
        ('x_on_cube', float, num_parameters),
        ('sim_id', int),
        ('local_pt', bool),
        ('local_min', bool),
  ]

    # State the generating function, its arguments, output, and necessary parameters.
    gen_specs = {
        'gen_f': gen_f,
        'in': ['x', 'f', 'local_pt', 'sim_id', 'returned', 'x_on_cube', 'local_min'],
        #'mu':0.1,   # limit on dist_to_bound: everything closer to bound than mu is thrown out
        'out': gen_out,
        'user':{
            'lb': lb,
            'ub': ub,
            'initial_sample_size': 20,  # num points sampled before starting opt runs, one per worker
            'localopt_method': 'scipy_COBYLA',
            'xatol':1e-10,
            'fatol':1e-10,
            'num_pts_first_pass': nworkers-1,
            'periodic': True,
        }
    }

    # Tell libEnsemble when to stop
    exit_criteria = {'sim_max': sim_max}

    persis_info = {'next_to_give': 0}
    persis_info['total_gen_calls'] = 0
    persis_info['last_worker'] = 0
    persis_info[0] = {
        'active_runs': set(),
        'run_order': {},
        'old_runs': {},
        'total_runs': 0,
        'rand_stream': np.random.RandomState()
    }

    for i in range(1, MPI.COMM_WORLD.Get_size()):
        persis_info[i] = {'rand_stream': np.random.RandomState()}

    alloc_specs = {'alloc_f': alloc_f, 'out': [('given_back', bool)], 'user': {}}

    H, persis_info, flag = libE(
        sim_specs,
        gen_specs,
        exit_criteria,
        persis_info=persis_info,
        alloc_specs=alloc_specs)
    if MPI.COMM_WORLD.Get_rank() == 0:
        return (H, persis_info)


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
        default = "100",
        help = "number of iterations, default is 100")
    parser.add_argument(
        "--nnodes", type = int,
        default = "6",
        help = "number of nodes in the 3-regular graph")
    parser.add_argument(
        "-p", type = int,
        default = "10",
        help = "maximum depth to explore")
    parser.add_argument(
        "--graph-generator-seed", type = int,
        default = "1",
        help = "seed for random graph generation")
    args = parser.parse_args()

    # generate objw
    # pass to libE

    #G = nx.random_regular_graph(3, args.nnodes, seed=args.graph_generator_seed)

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

    #lb = np.array([-np.inf, -np.inf] * args.p)
    #ub = np.array([np.inf, np.inf] * args.p)

    obj_w = ObjectiveWrapper(
            obj_f_cut, 
            varform_description={'name':'QAOA', 'p':args.p, 'cost_operator':C, 'num_qubits':G.number_of_nodes()}, 
            backend_description={'package':'qiskit', 'provider':'Aer', 'name':'statevector_simulator'},
            execute_parameters={})

    assert(obj_w.num_parameters == 2*args.p)

    t = optimize_obj(obj_w.get_obj(), obj_w.num_parameters, ub=ub, lb=lb, sim_max=args.maxiter)

    if MPI.COMM_WORLD.Get_rank() == 0:
        #outpath = f"/zfs/safrolab/users/rshaydu/quantum/data/nasa_2020/libe_optimized_schedules/n_{args.nnodes}_p_{args.p}_gseed_{args.graph_generator_seed}.p"
        outpath = f"/zfs/safrolab/users/rshaydu/quantum/data/nasa_2020/libe_optimized_schedules/petersen_p_{args.p}.p"
        print(f"Found solution {min(t[0]['f'])}, saving to {outpath}")
        pickle.dump(t, open(outpath, "wb"))
        
