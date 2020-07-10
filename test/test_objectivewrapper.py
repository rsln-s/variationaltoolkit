import unittest
import numpy as np
import networkx as nx
from functools import partial
from variationaltoolkit.objectivewrapper import ObjectiveWrapper
from variationaltoolkit.objectives import maxcut_obj
from variationaltoolkit.utils import brute_force, state_to_ampl_counts
from operator import itemgetter
from collections import Counter
from qiskit.optimization.applications.ising.max_cut import get_operator as get_maxcut_operator

import importlib.util
import sys
# a recipe for conditional import from https://docs.python.org/3/library/importlib.html#checking-if-a-module-can-be-imported
_mpsspec = importlib.util.find_spec('mpsbackend')

skip_mpsbackend = ('mpsbackend' not in sys.modules) and (_mpsspec is None)

class TestObjectiveWrapper(unittest.TestCase):

    def setUp(self):
        self.varform_description = {'name':'RYRZ', 'num_qubits':4, 'depth':1}
        self.backend_description={'package':'mpsbackend'}
        self.execute_parameters={'shots':100}
        self.w = np.array([[0,1,1,0],[1,0,1,1],[1,1,0,1],[0,1,1,0]])
        self.obj = partial(maxcut_obj, w=self.w) 
        self.C, _ = get_maxcut_operator(self.w)


    @unittest.skipIf(skip_mpsbackend, "mpsbackend not found")
    def test_get_obj(self):
        obj_w = ObjectiveWrapper(self.obj, varform_description=self.varform_description, backend_description=self.backend_description, execute_parameters=self.execute_parameters)
        obj_f = obj_w.get_obj()
        parameters = np.random.uniform(0, np.pi, obj_w.num_parameters)
        val = obj_f(parameters)
        self.assertIsInstance(val, float)

    def test_sv_obj(self):
        obj_w = ObjectiveWrapper(self.obj, 
                varform_description={'name':'QAOA', 'p':10, 'num_qubits':4, 'cost_operator':self.C}, 
                backend_description={'package':'qiskit', 'provider':'Aer', 'name':'statevector_simulator'}, 
                execute_parameters={})
        obj_f = obj_w.get_obj()
        parameters = np.array([ 5.97337687,  2.58355601,  1.40698116,  1.41929411, -0.78430107,
        -4.46418963, -0.61290647, -0.59975086,  0.48811492,  4.20269641,
        -2.71558857,  2.82117292,  2.93922949,  2.06076731,  2.19543793,
         2.42960372, -1.0079554 ,  2.22741002, -1.06316475,  0.53106839]) 
        val = obj_f(parameters)
        self.assertTrue(np.isclose(-3.935, val, atol=0.01))


    def test_qasm_sv_obj_custom_obj(self):
        G = nx.from_numpy_matrix(self.w)
        def obj_f_cut(x):
            cut = 0
            for i, j in G.edges():
                if x[i] != x[j]:
                    # the edge is cut
                    cut -= 1
            return cut
        obj_sv = ObjectiveWrapper(self.obj, 
                varform_description={'name':'QAOA', 'p':10, 'num_qubits':4, 'cost_operator':self.C}, 
                backend_description={'package':'qiskit', 'provider':'Aer', 'name':'statevector_simulator'}, 
                execute_parameters={}).get_obj()
        obj_qasm = ObjectiveWrapper(self.obj, 
                varform_description={'name':'QAOA', 'p':10, 'num_qubits':4, 'cost_operator':self.C}, 
                backend_description={'package':'qiskit', 'provider':'Aer', 'name':'qasm_simulator'}, 
                execute_parameters={'shots':10000}).get_obj()
        obj_sv_custom = ObjectiveWrapper(obj_f_cut, 
                varform_description={'name':'QAOA', 'p':10, 'num_qubits':4, 'cost_operator':self.C}, 
                backend_description={'package':'qiskit', 'provider':'Aer', 'name':'statevector_simulator'}, 
                execute_parameters={}).get_obj()
        obj_qasm_custom = ObjectiveWrapper(obj_f_cut, 
                varform_description={'name':'QAOA', 'p':10, 'num_qubits':4, 'cost_operator':self.C}, 
                backend_description={'package':'qiskit', 'provider':'Aer', 'name':'qasm_simulator'}, 
                execute_parameters={'shots':10000}).get_obj()
        parameters = np.array([ 5.97337687,  2.58355601,  1.40698116,  1.41929411, -0.78430107,
        -4.46418963, -0.61290647, -0.59975086,  0.48811492,  4.20269641,
        -2.71558857,  2.82117292,  2.93922949,  2.06076731,  2.19543793,
         2.42960372, -1.0079554 ,  2.22741002, -1.06316475,  0.53106839]) 
        sv_imported = obj_sv(parameters)
        qasm_imported = obj_qasm(parameters)
        sv_custom = obj_sv_custom(parameters)
        qasm_custom = obj_qasm_custom(parameters)
        self.assertTrue(np.isclose(sv_imported, sv_custom))
        self.assertTrue(np.isclose(sv_imported, qasm_imported, rtol=0.01))
        self.assertTrue(np.isclose(sv_custom, qasm_custom, rtol=0.01))


    def test_qasm_sv_obj_from_elist(self):
        elist = [[3,1],[3,2],[0,1],[0,2],[1,2]]
        G = nx.OrderedGraph()
        G.add_edges_from(elist)
        def obj_f_cut(x):
            cut = 0
            for i, j in G.edges():
                if x[i] != x[j]:
                    # the edge is cut
                    cut -= 1
            return cut
        w = nx.adjacency_matrix(G, nodelist=range(4)).toarray()
        obj = partial(maxcut_obj,w=w)
        C, _ = get_maxcut_operator(w)
        obj_sv = ObjectiveWrapper(obj, 
                varform_description={'name':'QAOA', 'p':10, 'num_qubits':4, 'cost_operator':C}, 
                backend_description={'package':'qiskit', 'provider':'Aer', 'name':'statevector_simulator'}, 
                execute_parameters={}).get_obj()
        obj_qasm = ObjectiveWrapper(obj, 
                varform_description={'name':'QAOA', 'p':10, 'num_qubits':4, 'cost_operator':C}, 
                backend_description={'package':'qiskit', 'provider':'Aer', 'name':'qasm_simulator'}, 
                execute_parameters={'shots':10000}).get_obj()
        obj_sv_custom = ObjectiveWrapper(obj_f_cut, 
                varform_description={'name':'QAOA', 'p':10, 'num_qubits':4, 'cost_operator':C}, 
                backend_description={'package':'qiskit', 'provider':'Aer', 'name':'statevector_simulator'}, 
                execute_parameters={}).get_obj()
        obj_qasm_custom = ObjectiveWrapper(obj_f_cut, 
                varform_description={'name':'QAOA', 'p':10, 'num_qubits':4, 'cost_operator':C}, 
                backend_description={'package':'qiskit', 'provider':'Aer', 'name':'qasm_simulator'}, 
                execute_parameters={'shots':10000}).get_obj()
        parameters = np.array([ 5.97337687,  2.58355601,  1.40698116,  1.41929411, -0.78430107,
        -4.46418963, -0.61290647, -0.59975086,  0.48811492,  4.20269641,
        -2.71558857,  2.82117292,  2.93922949,  2.06076731,  2.19543793,
         2.42960372, -1.0079554 ,  2.22741002, -1.06316475,  0.53106839]) 
        sv_imported = obj_sv(parameters)
        qasm_imported = obj_qasm(parameters)
        sv_custom = obj_sv_custom(parameters)
        qasm_custom = obj_qasm_custom(parameters)
        self.assertTrue(np.isclose(sv_imported, sv_custom))
        self.assertTrue(np.isclose(sv_imported, qasm_imported, rtol=0.01))
        self.assertTrue(np.isclose(sv_custom, qasm_custom, rtol=0.01))


    def test_qasm_sv_obj_peterson(self):
        elist = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0], [0, 5], [1, 6], [2, 7], [3, 8], [4, 9], [5, 7], [5, 8], [6, 8], [6, 9], [7, 9]]
        G = nx.OrderedGraph()
        G.add_edges_from(elist)
        def obj_f_cut(x):
            cut = 0
            for i, j in G.edges():
                if x[i] != x[j]:
                    # the edge is cut
                    cut -= 1
            return cut
        w = nx.adjacency_matrix(G, nodelist=range(10)).toarray()
        obj = partial(maxcut_obj,w=w)
        C, _ = get_maxcut_operator(w)
        brute_force_opt_imported, _ = brute_force(obj, G.number_of_nodes())
        brute_force_opt_custom, _ = brute_force(obj_f_cut, G.number_of_nodes())
        self.assertEqual(brute_force_opt_imported, brute_force_opt_custom)
        obj_sv = ObjectiveWrapper(obj, 
                varform_description={'name':'QAOA', 'p':9, 'num_qubits':10, 'cost_operator':C}, 
                backend_description={'package':'qiskit', 'provider':'Aer', 'name':'statevector_simulator'}, 
                objective_parameters={'save_resstrs':True},
                execute_parameters={})
        obj_qasm = ObjectiveWrapper(obj, 
                varform_description={'name':'QAOA', 'p':9, 'num_qubits':10, 'cost_operator':C}, 
                backend_description={'package':'qiskit', 'provider':'Aer', 'name':'qasm_simulator'}, 
                objective_parameters={'save_resstrs':True},
                execute_parameters={'shots':10000})
        obj_sv_custom = ObjectiveWrapper(obj_f_cut, 
                varform_description={'name':'QAOA', 'p':9, 'num_qubits':10, 'cost_operator':C}, 
                backend_description={'package':'qiskit', 'provider':'Aer', 'name':'statevector_simulator'}, 
                objective_parameters={'save_resstrs':True},
                execute_parameters={})
        obj_qasm_custom = ObjectiveWrapper(obj_f_cut, 
                varform_description={'name':'QAOA', 'p':9, 'num_qubits':10, 'cost_operator':C}, 
                backend_description={'package':'qiskit', 'provider':'Aer', 'name':'qasm_simulator'}, 
                objective_parameters={'save_resstrs':True},
                execute_parameters={'shots':10000})
        parameters = np.array([5.192253984583296, 5.144373231492732, 5.9438949617723775, 5.807748946652058, 3.533458907810596, 6.006206583282401, 6.122313961527631, 6.218468942101044, 6.227704753217614, 0.3895570099244132, -0.1809282325810937, 0.8844522327007089, 0.7916086532373585, 0.21294534589417236, 0.4328896243354414, 0.8327451563500539, 0.7694639329585451, 0.4727893829336214]) 
        sv_imported = obj_sv.get_obj()(parameters)
        resstrs_sv_imported = list(sorted(state_to_ampl_counts(obj_sv.resstrs[0],eps=0.01).items(), key=itemgetter(0))) 
        qasm_imported = obj_qasm.get_obj()(parameters)
        resstrs_qasm_imported = list(("".join(str(x) for x in k),v) for k, v in Counter(tuple(x) for x in obj_qasm.resstrs[0]).items() if v > 5)
        sv_custom = obj_sv_custom.get_obj()(parameters)
        resstrs_sv_custom = list(sorted(state_to_ampl_counts(obj_sv_custom.resstrs[0],eps=0.01).items(), key=itemgetter(0))) 
        qasm_custom = obj_qasm_custom.get_obj()(parameters)
        resstrs_qasm_custom = list(("".join(str(x) for x in k),v) for k, v in Counter(tuple(x) for x in obj_qasm_custom.resstrs[0]).items() if v > 5)
        self.assertTrue(set(x[0] for x in resstrs_sv_imported) == set(x[0] for x in resstrs_sv_custom))
        self.assertTrue(set(x[0] for x in resstrs_sv_imported) == set(x[0] for x in resstrs_qasm_imported))
        self.assertTrue(set(x[0] for x in resstrs_sv_custom) == set(x[0] for x in resstrs_qasm_custom))
        self.assertTrue(np.isclose(sv_imported, brute_force_opt_custom,rtol=0.01))
        self.assertTrue(np.isclose(sv_imported, sv_custom))
        self.assertTrue(np.isclose(sv_imported, qasm_imported, rtol=0.01))
        self.assertTrue(np.isclose(sv_custom, qasm_custom, rtol=0.01))


if __name__ == '__main__':
    unittest.main()
