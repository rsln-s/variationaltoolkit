# mpirun -np 16 --ppn 16 python slowtest_performance_aposmm_only.py
# Should run in under 20 sec

import os
import sys
import numpy as np
import networkx as nx
import pickle
from functools import partial
from qiskit.optimization.ising.max_cut import get_operator as get_maxcut_operator
import scipy
from variationaltoolkit import VariationalQuantumOptimizerAPOSMM
from variationaltoolkit import VariationalQuantumOptimizerSequential

from mpi4py import MPI
is_master = (MPI.COMM_WORLD.Get_rank() == 0)

world_size = MPI.COMM_WORLD.Get_size()

if is_master:
    start_time_aposmm = MPI.Wtime() 

def maxcut_obj(x,G):
    cut = 0
    for i, j in G.edges():
        if x[i] != x[j]:
            # the edge is cut
            cut -= 1
    return cut

elist = [
    [0,1],[1,2],[2,3],[3,4],[4,0],
    [0,5],[1,6],[2,7],[3,8],[4,9],
    [5,7],[5,8],[6,8],[6,9],[7,9]
]

G=nx.OrderedGraph()
G.add_edges_from(elist)

w = nx.adjacency_matrix(G)
obj = partial(maxcut_obj, G=G)
C, offset = get_maxcut_operator(w)

p = 10
lb = np.array([0, 0] * p)
ub = np.array([np.pi / 2] * p + [np.pi] * p)

np.random.seed(0)
init_theta = np.random.uniform(lb, ub)

np.random.seed(0)
varopt_aposmm = VariationalQuantumOptimizerAPOSMM(
        obj, 
        'scipy_COBYLA', 
        initial_point=init_theta,
        gen_specs_user={'max_active_runs': world_size-2},
        optimizer_parameters={'tol': 1e-10, 'options': {'disp':False, 'maxiter': 200}}, 
        varform_description={'name':'QAOA', 'p':p, 'cost_operator':C, 'num_qubits':G.number_of_nodes()}, 
        backend_description={'package':'qiskit', 'provider':'Aer', 'name':'statevector_simulator'}, 
        problem_description={'offset': offset, 'do_not_check_cost_operator':True},
        execute_parameters={})
res_aposmm = varopt_aposmm.optimize()
sys.stdout.flush()
sys.stderr.flush()
MPI.COMM_WORLD.Barrier()
if is_master:
    end_time_aposmm = MPI.Wtime()
    running_time = end_time_aposmm-start_time_aposmm
    print(f"APOSMM finished in {running_time}s with {world_size} processes", flush=True)
    assert(running_time < 25)
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    print(f"{script_name} finished successfully")
MPI.COMM_WORLD.Barrier()
