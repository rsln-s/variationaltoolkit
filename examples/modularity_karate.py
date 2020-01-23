import numpy as np
import networkx as nx
import argparse
import plotille 
from functools import partial
from variationaltoolkit import VariationalQuantumOptimizer
from variationaltoolkit.objectives import modularity_obj

# modularity maximization for karate club graph 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--optimizer", type = str,
        default = "SequentialOptimizer",
        help = "the optimization method, default is COBYLA")
    parser.add_argument(
        "--depth", type = int,
        default = "3",
        help = "the depth of the circuit, default is 3")
    parser.add_argument(
        "--shots", type = int,
        default = "10000",
        help = "the number of shots, default is 10000")
    parser.add_argument(
        "--maxiter", type = int,
        default = "100",
        help = "number of iterations, default is 100")
    
    args = parser.parse_args()

    G = nx.karate_club_graph()
    for node in G.nodes():
        G.nodes[node]['volume'] = G.degree[node]
    for u, v in G.edges():
        G[u][v]['weight'] = 1
    
    node_list = list(G.nodes())
    varform_description = {'package': 'mpsbackend', 'name': 'RYRZ', 'num_qubits': 68, 'depth': args.depth, 'entanglement': 'linear'}
    backend_description={'package': 'mpsbackend'}
    execute_parameters={'shots': args.shots}
    #optimizer_parameters={'maxiter': args.maxiter, 'disp':True}
    optimizer_parameters={'maxiter': args.maxiter}
    obj = partial(modularity_obj, N = 2, G = G, node_list = node_list)
    
    varopt = VariationalQuantumOptimizer(
            obj, 
            args.optimizer, 
            optimizer_parameters = optimizer_parameters, 
            varform_description = varform_description, 
            backend_description = backend_description, 
            execute_parameters = execute_parameters)
    varopt.optimize()
    res = varopt.get_optimal_solution(shots = args.shots)
    objs = varopt.obj_w.vals_statistic
    print(plotille.plot(range(len(objs)), objs))
