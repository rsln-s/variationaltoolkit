import unittest
import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from variationaltoolkit.optimizers import BOBYQA
from variationaltoolkit.utils import allclose_periodical

class TestBOBYQA(unittest.TestCase):

    def setUp(self):
        # TODO should use statevector simulator
        self.backend = Aer.get_backend('statevector_simulator')

    def test_optimize(self):
        N = 5
        tol = 1e-5
        def f(theta):
            qc = QuantumCircuit(N,N)
            for i in range(N):
                qc.rx(theta[i], i)
            sv = execute(qc, self.backend).result().get_statevector() 
            return np.linalg.norm(sv[1:]) 
        optimizer = BOBYQA(max_evals=1000, xtol_rel=tol)
        bounds = [(-np.pi, np.pi)] * N 
        opt_params, opt_val, num_optimizer_evals = optimizer.optimize(N, f, initial_point=np.full(N, np.pi/4), variable_bounds = bounds)
        print(opt_params, opt_val)
        self.assertTrue(np.isclose(f(opt_params), 0, atol=tol*10)) 
        self.assertTrue(allclose_periodical(opt_params, np.zeros(N), 0, 2*np.pi, atol=tol*10))


if __name__ == '__main__':
    unittest.main()
