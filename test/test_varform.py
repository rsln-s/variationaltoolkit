import unittest
from variationaltoolkit import VarForm
from qiskit.aqua.components.variational_forms import RYRZ

class TestVarForm(unittest.TestCase):

    def test_import_ryrz(self):
        var_form = VarForm(varform_description={'name':'RYRZ', 'num_qubits':5}, backend_description={})
        self.assertIsInstance(var_form.var_form, RYRZ)
        


# TODO:

# check that runs on mpsbackend and returns correct number of resstrs
# check that runs on qiskit.Aer qasm_simulator and returns correct number of resstrs
# check that runs on qiskit.Aer mps simulator and returns correct number of resstrs
# check that submits a job to IBMQ correctly

if __name__ == '__main__':
    unittest.main()
