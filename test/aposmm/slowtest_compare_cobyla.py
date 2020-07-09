import numpy as np
import networkx as nx
import pickle
from functools import partial
from qiskit.optimization.applications.ising.max_cut import get_operator as get_maxcut_operator
import scipy
from variationaltoolkit import VariationalQuantumOptimizerAPOSMM
from variationaltoolkit import VariationalQuantumOptimizerSequential

from mpi4py import MPI
is_master = (MPI.COMM_WORLD.Get_rank() == 0)

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
print(f"Init point: {init_theta}")

np.random.seed(0)
varopt_aposmm = VariationalQuantumOptimizerAPOSMM(
        obj, 
        'scipy_COBYLA', 
        initial_point=init_theta,
        optimizer_parameters={'tol': 1e-10, 'options': {'disp':False, 'maxiter': 300}}, 
        varform_description={'name':'QAOA', 'p':p, 'cost_operator':C, 'num_qubits':G.number_of_nodes()}, 
        backend_description={'package':'qiskit', 'provider':'Aer', 'name':'statevector_simulator'}, 
        problem_description={'offset': offset, 'do_not_check_cost_operator':True},
        execute_parameters={})
res_aposmm = varopt_aposmm.optimize()

if is_master:
    print(res_aposmm['min_val'])
    print(res_aposmm['opt_params'])
    np.random.seed(0)
    varopt_seq = VariationalQuantumOptimizerSequential(
            obj, 
            'COBYLA', 
            initial_point=init_theta,
            optimizer_parameters={'maxiter':300}, 
            varform_description={'name':'QAOA', 'p':p, 'cost_operator':C, 'num_qubits':G.number_of_nodes()}, 
            backend_description={'package':'qiskit', 'provider':'Aer', 'name':'statevector_simulator'}, 
            problem_description={'offset': offset, 'do_not_check_cost_operator':True},
            execute_parameters={})
    res_seq = varopt_seq.optimize()
    print("Both should be below -10!")
    print(res_seq['min_val'])
    print(res_seq['opt_params'])
    #print(res_aposmm['H']['x'])
    #print(varopt_seq.obj_w.points)

