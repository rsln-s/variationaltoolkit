import numpy as np
from functools import partial
from variationaltoolkit.objectives import maxcut_obj
from variationaltoolkit import VariationalQuantumOptimizer
import networkx as nx

N = 100
G = nx.erdos_renyi_graph(N, p=0.5)

varform_description = {'name':'RYRZ', 'num_qubits':N, 'depth':3, 'entanglement':'linear'}
backend_description={'package':'mpsbackend'}
execute_parameters={'shots':1000}
optimizer_parameters={'maxiter':100}
obj = partial(maxcut_obj, w=nx.adjacency_matrix(G)) 

import logging; logging.basicConfig(level=logging.INFO)
varopt = VariationalQuantumOptimizer(
        obj, 
        'COBYLA', 
        optimizer_parameters=optimizer_parameters, 
        varform_description=varform_description, 
        backend_description=backend_description, 
        execute_parameters=execute_parameters)
print(varopt.optimize())
res = varopt.get_optimal_solution()
print(res)
