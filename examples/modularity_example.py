import numpy as np
import networkx as nx
from functools import partial
from variationaltoolkit import VariationalQuantumOptimizerSequential
from variationaltoolkit.objectives import modularity_obj

w = np.array([[0,1,1,0,0,0],[1,0,1,0,0,0],[1,1,0,1,0,0],[0,0,1,0,1,1],[0,0,0,1,0,1],[0,0,0,1,1,0]])
G = nx.from_numpy_matrix(w)
B = nx.modularity_matrix(G, nodelist = list(range(6)))
m = G.number_of_edges()

varform_description = {'name':'RYRZ', 'num_qubits':6, 'depth':3, 'entanglement':'linear'}
backend_description={'package':'mpsbackend'}
execute_parameters={'shots':10000}
optimizer_parameters={'maxiter':10}
obj = partial(modularity_obj, N = 1, B = B, m = m)
    
varopt = VariationalQuantumOptimizerSequential(
         obj, 
         'COBYLA', 
         optimizer_parameters=optimizer_parameters, 
         varform_description=varform_description, 
         backend_description=backend_description, 
         execute_parameters=execute_parameters)
varopt.optimize()
res = varopt.get_optimal_solution(shots=10000)
x = res[1]
print(res)
