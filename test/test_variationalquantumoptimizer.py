import unittest
import numpy as np
import networkx as nx
from functools import partial
from variationaltoolkit.objectives import maxcut_obj, modularity_obj
from variationaltoolkit import VariationalQuantumOptimizer

import importlib.util
import sys
# a recipe for conditional import from https://docs.python.org/3/library/importlib.html#checking-if-a-module-can-be-imported
_mpsspec = importlib.util.find_spec('mpsbackend')

skip_mpsbackend = ('mpsbackend' not in sys.modules) and (_mpsspec is None)

class TestVariationalQuantumOptimizer(unittest.TestCase):

    def setUp(self):
        self.varform_description = {'name':'RYRZ', 'num_qubits':4, 'depth':3}
        self.backend_description={'package':'mpsbackend'}
        self.execute_parameters={'shots':1000}
        self.optimizer_parameters={'maxiter':50, 'disp':True}
        w = np.array([[0,1,1,0],[1,0,1,1],[1,1,0,1],[0,1,1,0]])
        self.obj = partial(maxcut_obj, w=w) 

    @unittest.skipIf(skip_mpsbackend, "mpsbackend not found")
    def test_maxcut(self):
        import logging; logging.disable(logging.CRITICAL)
        varopt = VariationalQuantumOptimizer(
                self.obj, 
                'COBYLA', 
                optimizer_parameters=self.optimizer_parameters, 
                varform_description=self.varform_description, 
                backend_description=self.backend_description, 
                execute_parameters=self.execute_parameters)
        varopt.optimize()
        res = varopt.get_optimal_solution()
        self.assertEqual(res[0], -4)
        self.assertTrue(np.array_equal(res[1], np.array([1,0,0,1])) or np.array_equal(res[1], np.array([0,1,1,0])))
        logging.disable(logging.NOTSET)

    @unittest.skipIf(skip_mpsbackend, "mpsbackend not found")
    def test_maxcut_seqopt(self):
        import logging; logging.disable(logging.CRITICAL)
        varopt = VariationalQuantumOptimizer(
                self.obj, 
                'SequentialOptimizer', 
                optimizer_parameters=self.optimizer_parameters, 
                varform_description=self.varform_description, 
                backend_description=self.backend_description, 
                execute_parameters=self.execute_parameters)
        varopt.optimize()
        res = varopt.get_optimal_solution()
        self.assertEqual(res[0], -4)
        self.assertTrue(np.array_equal(res[1], np.array([1,0,0,1])) or np.array_equal(res[1], np.array([0,1,1,0])))
        logging.disable(logging.NOTSET)

    @unittest.skipIf(skip_mpsbackend, "mpsbackend not found")
    def test_maxcut_mps_varform(self):
        import logging; logging.disable(logging.CRITICAL)
        varopt = VariationalQuantumOptimizer(
                self.obj, 
                'COBYLA', 
                optimizer_parameters=self.optimizer_parameters, 
                varform_description={'package':'mpsbackend', 'name':'RYRZ', 'num_qubits':4, 'depth':3}, 
                backend_description=self.backend_description, 
                execute_parameters=self.execute_parameters)
        varopt.optimize()
        res = varopt.get_optimal_solution()
        self.assertEqual(res[0], -4)
        self.assertTrue(np.array_equal(res[1], np.array([1,0,0,1])) or np.array_equal(res[1], np.array([0,1,1,0])))
        logging.disable(logging.NOTSET)

    def test_modularity(self):
        w = np.array([[0,1,1,0,0,0],[1,0,1,0,0,0],[1,1,0,1,0,0],[0,0,1,0,1,1],[0,0,0,1,0,1],[0,0,0,1,1,0]])
        G = nx.from_numpy_matrix(w)
        for node in G.nodes():
            G.nodes[node]['volume'] = G.degree[node]
        for u, v in G.edges():
            G[u][v]['weight'] = 1
        node_list = list(G.nodes())
        mod_obj = partial(modularity_obj, N = 1, G = G, node_list = node_list)
            
        varopt = VariationalQuantumOptimizer(
                 mod_obj, 
                 'SequentialOptimizer', 
                optimizer_parameters=self.optimizer_parameters, 
                varform_description={'name':'RYRZ', 'num_qubits':6, 'depth':3, 'entanglement':'linear'}, 
                backend_description={'package':'qiskit', 'provider':'Aer', 'name':'qasm_simulator'}, 
                execute_parameters=self.execute_parameters)
        varopt.optimize()
        res = varopt.get_optimal_solution(shots=10000)
        self.assertTrue(np.array_equal(res[1], np.array([0,0,0,1,1,1])) or np.array_equal(res[1], np.array([1,1,1,0,0,0])))
    
        
if __name__ == '__main__':
    unittest.main()
