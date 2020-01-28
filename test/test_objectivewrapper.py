import unittest
import numpy as np
from functools import partial
from variationaltoolkit.objectivewrapper import ObjectiveWrapper
from variationaltoolkit.objectives import maxcut_obj

import importlib.util
import sys
# a recipe for conditional import from https://docs.python.org/3/library/importlib.html#checking-if-a-module-can-be-imported
_mpsspec = importlib.util.find_spec('mpsbackend')

skip_mpsbackend = ('mpsbackend' not in sys.modules) and (_mpsspec is None)

@unittest.skipIf(skip_mpsbackend, "mpsbackend not found")
class TestObjectiveWrapper(unittest.TestCase):

    def setUp(self):
        self.varform_description = {'name':'RYRZ', 'num_qubits':4, 'depth':1}
        self.backend_description={'package':'mpsbackend'}
        self.execute_parameters={'shots':100}
        w = np.array([[0,1,1,0],[1,0,1,1],[1,1,0,1],[0,1,1,0]])
        self.obj = partial(maxcut_obj, w=w) 

    def test_get_obj(self):
        obj_w = ObjectiveWrapper(self.obj, varform_description=self.varform_description, backend_description=self.backend_description, execute_parameters=self.execute_parameters)
        obj_f = obj_w.get_obj()
        parameters = np.random.uniform(0, np.pi, obj_w.var_form.num_parameters)
        val = obj_f(parameters)
        self.assertIsInstance(val, float)


if __name__ == '__main__':
    unittest.main()
