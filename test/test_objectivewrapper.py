import unittest
import numpy as np
from functools import partial
from variationaltoolkit.objectivewrapper import ObjectiveWrapper
from variationaltoolkit.objectives import maxcut_obj
from qiskit.optimization.ising.max_cut import get_operator as get_maxcut_operator

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
        w = np.array([[0,1,1,0],[1,0,1,1],[1,1,0,1],[0,1,1,0]])
        self.obj = partial(maxcut_obj, w=w) 
        self.C, _ = get_maxcut_operator(w)



    @unittest.skipIf(skip_mpsbackend, "mpsbackend not found")
    def test_get_obj(self):
        obj_w = ObjectiveWrapper(self.obj, varform_description=self.varform_description, backend_description=self.backend_description, execute_parameters=self.execute_parameters)
        obj_f = obj_w.get_obj()
        parameters = np.random.uniform(0, np.pi, obj_w.var_form.num_parameters)
        val = obj_f(parameters)
        self.assertIsInstance(val, float)

    def test_sv_obj(self):
        obj_w = ObjectiveWrapper(self.obj, 
                varform_description={'name':'QAOA', 'p':10, 'num_qubits':4, 'cost_operator':self.C}, 
                backend_description={'package':'qiskit', 'provider':'Aer', 'name':'statevector_simulator'}, 
                objective_parameters={'num_processes':8},
                execute_parameters={})
        obj_f = obj_w.get_obj()
        parameters = np.array([ 5.97337687,  2.58355601,  1.40698116,  1.41929411, -0.78430107,
        -4.46418963, -0.61290647, -0.59975086,  0.48811492,  4.20269641,
        -2.71558857,  2.82117292,  2.93922949,  2.06076731,  2.19543793,
         2.42960372, -1.0079554 ,  2.22741002, -1.06316475,  0.53106839]) 
        val = obj_f(parameters)
        self.assertTrue(np.isclose(-3.935, val, atol=0.01))

if __name__ == '__main__':
    unittest.main()
