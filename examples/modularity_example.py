import numpy as np
import networkx as nx
from functools import partial
from variationaltoolkit import VariationalQuantumOptimizer
from variationaltoolkit.objectives import modularity_obj, bin_to_dec, compute_objective

w = np.array([[0,1,1,0,0,0],[1,0,1,0,0,0],[1,1,0,1,0,0],[0,0,1,0,1,1],[0,0,0,1,0,1],[0,0,0,1,1,0]])
G = nx.from_numpy_matrix(w)
for node in G.nodes():
    G.nodes[node]['volume'] = 1
for u, v in G.edges():
    G[u][v]['weight'] = 1
node_list = list(G.nodes())
varform_description = {'name':'RYRZ', 'num_qubits':6, 'depth':3, 'entanglement':'linear'}
backend_description={'package':'mpsbackend'}
execute_parameters={'shots':10000}
optimizer_parameters={'maxiter':10}
obj = partial(modularity_obj, N = 1, G = G, node_list = node_list)
    
varopt = VariationalQuantumOptimizer(
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