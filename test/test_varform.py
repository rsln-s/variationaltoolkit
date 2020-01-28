import unittest
import numpy as np
import importlib.util
import sys
from variationaltoolkit import VarForm
from qiskit.aqua.components.variational_forms import RYRZ
from qiskit.optimization.ising.max_cut import get_operator as get_maxcut_operator

# a recipe for conditional import from https://docs.python.org/3/library/importlib.html#checking-if-a-module-can-be-imported
_mpsspec = importlib.util.find_spec('mpsbackend')

skip_mpsbackend = ('mpsbackend' not in sys.modules) and (_mpsspec is None)

class TestVarForm(unittest.TestCase):

    def setUp(self):
        self.varform_description = {'name':'RYRZ', 'num_qubits':5, 'depth':1}

    def test_import_ryrz(self):
        var_form = VarForm(varform_description=self.varform_description)
        self.assertIsInstance(var_form.var_form, RYRZ)
        
    def test_ryrz_qasm_simulator(self):
        var_form = VarForm(varform_description=self.varform_description)
        parameters = np.random.uniform(0, np.pi, var_form.num_parameters)
        execute_parameters={'shots':100}
        resstrs = var_form.run(parameters, 
                backend_description={'package':'qiskit', 'provider':'Aer', 'name':'qasm_simulator'},
                execute_parameters=execute_parameters)
        self.assertEqual(len(resstrs), execute_parameters['shots'])
        self.assertTrue(all(len(x) == self.varform_description['num_qubits'] for x in resstrs))

    def test_qaoa_maxcut(self):
        w = np.array([[0,1,1,0],[1,0,1,1],[1,1,0,1],[0,1,1,0]])
        C, offset = get_maxcut_operator(w)
        var_form = VarForm(varform_description={'name':'QAOA', 'p':2, 'cost_operator':C, 'num_qubits':4})
        parameters = np.random.uniform(0, np.pi, var_form.num_parameters)
        execute_parameters={'shots':100}
        resstrs = var_form.run(parameters, 
                backend_description={'package':'qiskit', 'provider':'Aer', 'name':'qasm_simulator'},
                execute_parameters=execute_parameters)
        self.assertEqual(len(resstrs), execute_parameters['shots'])
        self.assertTrue(all(len(x) == 4 for x in resstrs))

    @unittest.skipIf(skip_mpsbackend, "mpsbackend not found")
    def test_ryrz_mpssimulator(self):
        var_form = VarForm(varform_description=self.varform_description)
        parameters = np.random.uniform(0, np.pi, var_form.num_parameters)
        execute_parameters={'shots':100}
        resstrs = var_form.run(parameters, 
                backend_description={'package':'mpsbackend'},
                execute_parameters=execute_parameters)
        self.assertEqual(len(resstrs), execute_parameters['shots'])
        self.assertTrue(all(len(x) == self.varform_description['num_qubits'] for x in resstrs))

# TODO:

# check that runs on qiskit.Aer mps simulator and returns correct number of resstrs
# check that submits a job to IBMQ correctly

if __name__ == '__main__':
    unittest.main()
