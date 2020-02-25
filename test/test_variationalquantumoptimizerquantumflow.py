import unittest
import numpy as np
import networkx as nx
import time
from functools import partial
from qiskit.optimization.ising.max_cut import get_operator as get_maxcut_operator
from variationaltoolkit.objectives import maxcut_obj, modularity_obj
from variationaltoolkit import VariationalQuantumOptimizerQuantumFlow

import importlib.util
import sys
# a recipe for conditional import from https://docs.python.org/3/library/importlib.html#checking-if-a-module-can-be-imported
_mpsspec = importlib.util.find_spec('mpsbackend')

skip_mpsbackend = ('mpsbackend' not in sys.modules) and (_mpsspec is None)

class TestVariationalQuantumOptimizerQuantumFlow(unittest.TestCase):

    def setUp(self):
        self.w = np.array([[0,1,1,0],[1,0,1,1],[1,1,0,1],[0,1,1,0]])
        self.obj = partial(maxcut_obj, w=self.w) 

    def test_maxcut_qaoa(self):
        # TODO This is just a stub. Modify as needed. Also add more tests
        import logging; logging.disable(logging.CRITICAL)
        C, offset = get_maxcut_operator(self.w)
        varopt = VariationalQuantumOptimizerQuantumFlow(
                self.obj, 
                'AdamOptimizer', 
                varform_description={'name':'QAOA', 'p':10, 'cost_operator':C, 'num_qubits':4}, 
                problem_description={'offset': offset})
        res = varopt.optimize()
        self.assertTrue(np.isclose(res['min_val'],-4))
        logging.disable(logging.NOTSET)

        
if __name__ == '__main__':
    unittest.main()
