# K2,3 p = 1, see if 00011 state prob > 95%
# K2,3 p = 2, see if 11100 state prob > 1% (this will take a while ~10 or 20 sec)

# K2,3 p = 1, initial_state = '00000', see if 00011 state prob > 95%
# K2,3 p = 1, initial_state = '11100', see if 11100 state prob > 95%

# K2,3 p = 1, initial_state = '00111', see if error is throwed

# K3,2 p = 1, see if 00111 state prob > 85%
# K5,5 p = 1, see if 0000011111 state prob > 95%

from variationaltoolkit.stuart_mis_utils.stuartansatzfunctions import stuart_one_run, stuart_compute_energy_avg, stuart_compute_energy_average_min

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

class TestMIS(unittest.TestCase):

        
    """
    K2,3 p = 1, see if 00011 state prob > 85%
    K2,3 p = 2, see if 11100 state prob > 1% (this will take a while ~10 or 20 sec)
    """
    def test_stuart_ansatz(self):
        # K2,3 graph
        elist = [[0,2],[0,3],[0,4],[1,2],[1,3],[1,4]]
        G = nx.OrderedGraph()
        G.add_edges_from(elist)
        
        counts_1, res_1 = stuart_one_run(1, G)
        counts_2, res_2 = stuart_one_run(2, G)
        self.assertTrue((counts_1['00011'] / sum(counts_1.values())) > 0.85)
        self.assertTrue((counts_2['11100'] / sum(counts_2.values())) > 0.01)
    
    
    """
    K2,3 p = 1, initial_state = '00000', see if 00011 state prob > 85%
    K2,3 p = 1, initial_state = '11100', see if 11100 state prob > 95%
    """
    def test_stuart_ansatz_initial_state(self):
        # K2,3 graph
        elist = [[0,2],[0,3],[0,4],[1,2],[1,3],[1,4]]
        G = nx.OrderedGraph()
        G.add_edges_from(elist)
        
        counts_1, res_1 = stuart_one_run(1, G, initial_state_string='00000')
        counts_2, res_2 = stuart_one_run(1, G, initial_state_string='11100')
        self.assertTrue((counts_1['00011'] / sum(counts_1.values())) > 0.85)
        self.assertTrue((counts_2['11100'] / sum(counts_2.values())) > 0.90)
    
    
    """
    K2,3 p = 1, initial_state = '00111', see if error is throwed
    """
    def test_stuart_ansatz_nonIS_error_catch(self):
        # K2,3 graph
        elist = [[0,2],[0,3],[0,4],[1,2],[1,3],[1,4]]
        G = nx.OrderedGraph()
        G.add_edges_from(elist)
        try:
            stuart_compute_energy_avg(1, G, initial_state_string='00111') # the input state is not a valid independent set
        except ValueError as e:
            pass
        else:
            self.fail("Function did not raise error!")
    
    
    """
    K3,2 p = 1, see if 00111 state prob > 85%
    """
    def test_stuart_ansatz_different_order(self):
        # K3,2 graph
        elist = [[0,3],[0,4],[1,3],[1,4],[2,3],[2,4]]
        G = nx.OrderedGraph()
        G.add_edges_from(elist)
        
        counts, res = stuart_one_run(1, G)
        self.assertTrue((counts['00111'] / sum(counts.values())) > 0.85)
        
    
    """
    K5,5 p = 1, see if 0000011111 state prob > 95%
    """
    def test_stuart_ansatz_larger_symmetric(self):
        #Bipartite 5,5 d=3 regular graph
        elist = [[0,5],[0,6],[0,7],[1,6],[1,7],[1,8],[2,7],[2,8],[2,9],[3,8],[3,9],[3,5],[4,5],[4,6],[4,9]]
        G = nx.OrderedGraph()
        G.add_edges_from(elist)
        
        counts, res = stuart_one_run(1, G)
        self.assertTrue((counts['0000011111'] / sum(counts.values())) > 0.95)
            
    
    """
    Edge case: K2,3 with one isolated node
    """
    def test_stuart_ansatz_isolated_node(self):
        import networkx as nx
        elist = [[0,2],[0,3],[0,4],[1,2],[1,3],[1,4]]
        G = nx.OrderedGraph()
        G.add_edges_from(elist)
        G.add_node(5)
        
        counts, res = stuart_one_run(1, G)
        self.assertTrue((counts['100011'] / sum(counts.values())) > 0.85)
        
        
if __name__ == '__main__':
    unittest.main()
