import unittest
import numpy as np
import networkx as nx
import time
from functools import partial
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.optimization.ising.max_cut import get_operator as get_maxcut_operator
from variationaltoolkit.objectives import maxcut_obj, modularity_obj
from variationaltoolkit import VariationalQuantumOptimizerSequential
from variationaltoolkit.objectives import modularity_energy

import importlib.util
import sys
# a recipe for conditional import from https://docs.python.org/3/library/importlib.html#checking-if-a-module-can-be-imported
_mpsspec = importlib.util.find_spec('mpsbackend')

skip_mpsbackend = ('mpsbackend' not in sys.modules) and (_mpsspec is None)

class TestVariationalQuantumOptimizerSequential(unittest.TestCase):

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
        varopt = VariationalQuantumOptimizerSequential(
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
        varopt = VariationalQuantumOptimizerSequential(
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
        varopt = VariationalQuantumOptimizerSequential(
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

    @unittest.skipIf(skip_mpsbackend, "mpsbackend not found")
    def test_mpo_objective_modularity_k_way(self):
        import logging; logging.disable(logging.CRITICAL)
        import matlab
        G = nx.OrderedGraph()
        elist = [(0, 2), (0, 3), (0, 11), (1, 2), (1, 3), (2, 3), (3, 4), (4, 6), (4, 7), (5, 6), (5, 7), (6, 7), (7, 8), (8, 10), (8, 11), (9, 10), (9, 11), (10, 11)]
        G.add_edges_from(elist)
        nnodes = G.number_of_nodes()
        N = 2
        nqubits = nnodes * N # N variables per node -- up to 2**N communities

        node_list = list(range(nnodes))
        B = matlab.double(nx.modularity_matrix(G, nodelist=node_list).tolist())

        obj = partial(modularity_energy, 
                N = N, 
                B = nx.modularity_matrix(G, nodelist=node_list).getA(), 
                m = G.number_of_edges(), 
                deg_list = [G.degree[i] for i in node_list])

        precomputed_good_params = np.array([ 4.15610651,  1.31952566,  5.39528713,  0.03165722,  0.29017244,
        1.13614827, -2.07509482,  2.29757058,  0.41243041,  5.76625946,
       -4.32125694, -2.21142374, -2.95591283, -0.44569759,  5.00991472,
       -2.38284874,  4.08231081,  1.55265289, -5.20439166, -2.62705216,
        0.26832527, -4.12101477, -4.27473955,  4.58479753, -1.56014587,
       -0.52379135, -2.78755743,  6.35824237,  6.16095312, -4.80030703,
       -0.84348438, -2.9339378 , -3.13894666,  3.30636124, -3.2208338 ,
       -4.00239323,  1.91003078,  5.73582487,  1.4899889 ,  5.80934756,
        3.10322086,  1.60227976,  4.74931798,  2.32907816,  4.38702652,
       -4.80603026, -0.82120823, -3.2610527 , -3.64614703,  5.99653502,
        4.94231979,  3.73974001, -2.34897493, -4.71583219,  1.48330405,
        6.04077852,  2.84132271,  0.01466078, -3.61453391, -5.68864954,
        6.08686394,  5.46570721,  5.44826073, -5.26395504, -0.93564532,
        1.89588868, -5.21973893,  3.01863335,  2.51588271, -3.97664614,
        7.58394176, -3.59918101,  0.96797456, -0.74624669, -4.83039533,
       -4.03002175,  0.70923642, -3.75009256,  4.82231404, -4.22920321,
       -3.52680223,  0.31700832,  6.75879683,  3.50625919, -5.1009759 ,
        3.0347534 , -4.09756222, -3.42942231,  2.83041522, -3.9895508 ,
       -2.8693705 ,  2.12212776,  0.39196109,  7.48359308, -3.69241341,
       -3.46325887, -2.33739302, -5.40617378,  3.54184289,  4.31628083,
        0.29044417,  4.18564356,  2.30074315,  2.57942552, -3.10036929,
       -4.30000567, -1.62644728,  5.24134547, -1.74320921,  3.27039832,
        4.08539243,  4.56899165,  5.74862203, -2.40435231,  3.44914951,
       -5.03919062,  2.84952808, -1.13341506,  5.00855887, -5.66899901,
        0.28431306, -4.32562784,  2.91942835,  5.01386366,  0.36082722,
        7.66761106,  4.13723874, -4.45246101,  0.13413378,  5.94874616,
        5.92336011,  6.59555163, -3.31464691,  2.2468367 , -4.56588791,
        1.01536706,  0.34020607,  3.74764888,  1.65250476, -4.72223208,
        2.38389989, -6.36721914, -2.80402566, -4.48821116,  0.28952596,
       -4.51485203,  3.08119184, -0.60249566,  6.10183681,  5.66230155,
        0.30292948,  2.08435297,  2.48144112, -0.4033077 ,  3.18158093,
        4.38631354, -1.37507408, -0.72594662, -3.35534963, -0.09028156,
       -5.56034943,  3.40867727,  2.51204148, -3.50149819,  3.21263761,
        1.39083258, -0.3292894 , -2.45686952, -4.17826603, -3.08442966,
       -2.04319878,  1.13819382, -2.12785284, -0.62048513, -3.67792066,
        5.43649767, -3.14400277,  3.01728892, -0.62765206, -1.6561974 ,
        4.61813977,  5.65717505,  2.8793227 ,  0.67725344,  3.29505459,
        0.33112227,  3.07787261, -2.43315426, -1.11027068,  3.18688124,
        2.95611302, -1.09703103])

        varopt = VariationalQuantumOptimizerSequential(
                obj, 
                'COBYLA', 
                optimizer_parameters={'maxiter':5}, 
                initial_point = precomputed_good_params,
                varform_description={'package':'mpsbackend', 'name':'RYRZ', 'num_qubits':nqubits, 'depth':3, 'entanglement_gate':'cz'}, 
                backend_description=self.backend_description, 
                objective_parameters={'use_mpo_energy':True, 'hamiltonian_constructor': 'construct_k_way_modularity_Hamiltonian', 
                    'hamiltonian_constructor_parameters': [B, N]},
                execute_parameters={})
        optres = varopt.optimize()
        self.assertTrue(optres['min_val'] < -19)
        logging.disable(logging.NOTSET)

    def test_modularity(self):
        w = np.array([[0,1,1,0,0,0],[1,0,1,0,0,0],[1,1,0,1,0,0],[0,0,1,0,1,1],[0,0,0,1,0,1],[0,0,0,1,1,0]])
        G = nx.from_numpy_matrix(w)
        B = nx.modularity_matrix(G, nodelist = list(range(6)))
        m = G.number_of_edges()
        mod_obj = partial(modularity_obj, N = 1, B = B, m = m)
            
        varopt = VariationalQuantumOptimizerSequential(
                 mod_obj, 
                 'SequentialOptimizer', 
                optimizer_parameters=self.optimizer_parameters, 
                varform_description={'name':'RYRZ', 'num_qubits':6, 'depth':3, 'entanglement':'linear'}, 
                backend_description={'package':'qiskit', 'provider':'Aer', 'name':'qasm_simulator'}, 
                execute_parameters=self.execute_parameters)
        varopt.optimize()
        res = varopt.get_optimal_solution(shots=10000)
        self.assertTrue(np.array_equal(res[1], np.array([0,0,0,1,1,1])) or np.array_equal(res[1], np.array([1,1,1,0,0,0])))
    
    def test_modularity_sv(self):
        w = np.array([[0,1,1,0,0,0],[1,0,1,0,0,0],[1,1,0,1,0,0],[0,0,1,0,1,1],[0,0,0,1,0,1],[0,0,0,1,1,0]])
        G = nx.from_numpy_matrix(w)
        B = nx.modularity_matrix(G, nodelist = list(range(6)))
        m = G.number_of_edges()
        mod_obj = partial(modularity_obj, N = 1, B = B, m = m)
            
        varopt = VariationalQuantumOptimizerSequential(
                 mod_obj, 
                 'SequentialOptimizer', 
                optimizer_parameters=self.optimizer_parameters, 
                varform_description={'name':'RYRZ', 'num_qubits':6, 'depth':3, 'entanglement':'linear'}, 
                backend_description={'package':'qiskit', 'provider':'Aer', 'name':'statevector_simulator'}, 
                execute_parameters=self.execute_parameters)
        varopt.optimize()
        res = varopt.get_optimal_solution(shots=10000)
        self.assertTrue(np.array_equal(res[1], np.array([0,0,0,1,1,1])) or np.array_equal(res[1], np.array([1,1,1,0,0,0])))
    
    def test_maxcut_qaoa(self):
        import logging; logging.disable(logging.CRITICAL)
        C, offset = get_maxcut_operator(self.w)
        varopt = VariationalQuantumOptimizerSequential(
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
        varopt = VariationalQuantumOptimizerSequential(
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
        varopt = VariationalQuantumOptimizerSequential(
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
        varopt = VariationalQuantumOptimizerSequential(
                self.obj, 
                'COBYLA', 
                initial_point=[np.pi/4, 0, 0, np.pi/2],
                optimizer_parameters={'maxiter':100}, 
                varform_description={'name':'QAOA', 'p':15, 'cost_operator':C, 'num_qubits':4}, 
                backend_description={'package':'qiskit', 'provider':'Aer', 'name':'statevector_simulator'}, 
                problem_description={'offset': offset, 'smooth_schedule':True},
                execute_parameters=self.execute_parameters)
        res = varopt.optimize()
        print(res)
        self.assertTrue(res['min_val'] < -3.5)
        logging.disable(logging.NOTSET)
        
    def test_maxcut_qaoa_large(self):
        G = nx.random_regular_graph(3, 20, seed=1)
        w = nx.adjacency_matrix(G, nodelist=range(G.number_of_nodes()))
        obj = partial(maxcut_obj, w=w)
        C, offset = get_maxcut_operator(w)
        start = time.time()
        varopt = VariationalQuantumOptimizerSequential(
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
