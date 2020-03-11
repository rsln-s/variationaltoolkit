import unittest
import numpy as np
import networkx as nx
import time
from functools import partial
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.optimization.ising.max_cut import get_operator as get_maxcut_operator
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
        self.optimizer_parameters={'maxiter':50}
        self.w = np.array([[0,1,1,0],[1,0,1,1],[1,1,0,1],[0,1,1,0]])
        self.obj = partial(maxcut_obj, w=self.w) 

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
    
    def test_maxcut_qaoa(self):
        import logging; logging.disable(logging.CRITICAL)
        C, offset = get_maxcut_operator(self.w)
        varopt = VariationalQuantumOptimizer(
                self.obj, 
                'COBYLA', 
                optimizer_parameters=self.optimizer_parameters, 
                varform_description={'name':'QAOA', 'p':2, 'cost_operator':C, 'num_qubits':4}, 
                backend_description={'package':'qiskit', 'provider':'Aer', 'name':'qasm_simulator'}, 
                problem_description={'offset': offset},
                execute_parameters=self.execute_parameters)
        varopt.optimize()
        res = varopt.get_optimal_solution()
        self.assertEqual(res[0], -4)
        self.assertTrue(np.array_equal(res[1], np.array([1,0,0,1])) or np.array_equal(res[1], np.array([0,1,1,0])))
        logging.disable(logging.NOTSET)

    def test_maxcut_qaoa_sv(self):
        import logging; logging.disable(logging.CRITICAL)
        C, offset = get_maxcut_operator(self.w)
        varopt = VariationalQuantumOptimizer(
                self.obj, 
                'COBYLA', 
                optimizer_parameters=self.optimizer_parameters, 
                varform_description={'name':'QAOA', 'p':2, 'cost_operator':C, 'num_qubits':4}, 
                backend_description={'package':'qiskit', 'provider':'Aer', 'name':'statevector_simulator'}, 
                problem_description={'offset': offset},
                execute_parameters=self.execute_parameters)
        varopt.optimize()
        res = varopt.get_optimal_solution()
        self.assertEqual(res[0], -4)
        self.assertTrue(np.array_equal(res[1], np.array([1,0,0,1])) or np.array_equal(res[1], np.array([0,1,1,0])))
        logging.disable(logging.NOTSET)

    def test_maxcut_qaoa_mixer_circuit(self):
        import logging; logging.disable(logging.CRITICAL)
        C, offset = get_maxcut_operator(self.w)
        # build transverse field mixer circuit
        mixer_circuit = QuantumCircuit(4)
        beta = Parameter('beta')
        for q1 in range(4):
            mixer_circuit.h(q1)
            mixer_circuit.rz(2*beta, q1)
            mixer_circuit.h(q1)
        # pass it to variational quantum optimizer
        varopt = VariationalQuantumOptimizer(
                self.obj, 
                'COBYLA', 
                optimizer_parameters=self.optimizer_parameters, 
                varform_description={'name':'QAOA', 'p':2, 'cost_operator':C, 'num_qubits':4, 'use_mixer_circuit':True, 'mixer_circuit':mixer_circuit}, 
                backend_description={'package':'qiskit', 'provider':'Aer', 'name':'statevector_simulator'}, 
                problem_description={'offset': offset},
                execute_parameters=self.execute_parameters)
        varopt.optimize()
        res = varopt.get_optimal_solution()
        self.assertEqual(res[0], -4)
        self.assertTrue(np.array_equal(res[1], np.array([1,0,0,1])) or np.array_equal(res[1], np.array([0,1,1,0])))
        logging.disable(logging.NOTSET)

    def test_maxcut_qaoa_smooth(self):
        import logging; logging.disable(logging.CRITICAL)
        C, offset = get_maxcut_operator(self.w)
        varopt = VariationalQuantumOptimizer(
                self.obj, 
                'COBYLA', 
                initial_point=[np.pi/4, 0, 0, np.pi/2],
                optimizer_parameters=self.optimizer_parameters, 
                varform_description={'name':'QAOA', 'p':15, 'cost_operator':C, 'num_qubits':4}, 
                backend_description={'package':'qiskit', 'provider':'Aer', 'name':'statevector_simulator'}, 
                problem_description={'offset': offset, 'smooth_schedule':True},
                execute_parameters=self.execute_parameters)
        res = varopt.optimize()
        self.assertTrue(res['min_val'] < -3.5)
        logging.disable(logging.NOTSET)
        
    def test_maxcut_qaoa_large(self):
        G = nx.random_regular_graph(3, 20, seed=1)
        w = nx.adjacency_matrix(G, nodelist=range(G.number_of_nodes()))
        obj = partial(maxcut_obj, w=w)
        C, offset = get_maxcut_operator(w)
        start = time.time()
        varopt = VariationalQuantumOptimizer(
                obj, 
                'COBYLA', 
                initial_point=np.zeros(2),
                optimizer_parameters={'maxiter':1, 'disp':True}, 
                varform_description={'name':'QAOA', 'p':1, 'cost_operator':C, 'num_qubits':G.number_of_nodes()}, 
                backend_description={'package':'qiskit', 'provider':'Aer', 'name':'qasm_simulator'}, 
                problem_description={'offset': offset, 'do_not_check_cost_operator':True},
                execute_parameters=self.execute_parameters)
        res = varopt.optimize()
        end = time.time()
        self.assertTrue(end-start < 1)
        
if __name__ == '__main__':
    unittest.main()
