import numpy as np
from functools import partial
import variationaltoolkit
from variationaltoolkit.objectives import maxcut_obj
from variationaltoolkit import VariationalQuantumOptimizerSequential
import networkx as nx
import logging
from variationaltoolkit.utils import set_log_level
set_log_level(logging.INFO)

N = 10
G = nx.erdos_renyi_graph(N, p=0.5)

varform_description = {'name':'RYRZ', 'num_qubits':N, 'depth':3, 'entanglement':'linear'}
backend_description={'package':'qiskit', 'provider':'Aer', 'name':'qasm_simulator'}
execute_parameters={'shots':1000}
optimizer_parameters={'maxiter':10}
obj = partial(maxcut_obj, w=nx.adjacency_matrix(G)) 

varopt = VariationalQuantumOptimizerSequential(
        obj, 
        'COBYLA', 
        optimizer_parameters=optimizer_parameters, 
        varform_description=varform_description, 
        backend_description=backend_description, 
        execute_parameters=execute_parameters)
print(varopt.optimize())
res = varopt.get_optimal_solution()
print(res)
