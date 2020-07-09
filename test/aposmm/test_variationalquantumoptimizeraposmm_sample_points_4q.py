import numpy as np
import os
import networkx as nx
from functools import partial
from qiskit.optimization.applications.ising.max_cut import get_operator as get_maxcut_operator
from variationaltoolkit.objectives import maxcut_obj, modularity_obj
from variationaltoolkit import VariationalQuantumOptimizerAPOSMM

from mpi4py import MPI
is_master = (MPI.COMM_WORLD.Get_rank() == 0)

w = np.array([[0,1,1,0],[1,0,1,1],[1,1,0,1],[0,1,1,0]])
obj = partial(maxcut_obj, w=w) 
C, offset = get_maxcut_operator(w)
p = 4
maxiter = 20
initial_sample_size = 20
sample_points = np.split(np.linspace(0, np.pi, initial_sample_size * 2 * p), initial_sample_size)
varopt = VariationalQuantumOptimizerAPOSMM(
        obj, 
        'scipy_COBYLA', 
        gen_specs_user={'initial_sample_size':initial_sample_size, 'sample_points': sample_points},
        optimizer_parameters={'maxiter':maxiter}, 
        varform_description={'name':'QAOA', 'p':p, 'cost_operator':C, 'num_qubits':4}, 
        backend_description={'package':'qiskit', 'provider':'Aer', 'name':'statevector_simulator'}, 
        problem_description={'offset': offset},
        execute_parameters={})
res = varopt.optimize()
if is_master:
    assert(len(res['H']['x']) == maxiter)
    assert(np.allclose(res['H']['x'], sample_points))
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    print(f"{script_name} finished successfully")
