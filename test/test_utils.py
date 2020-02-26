import unittest
import numpy as np
import networkx as nx
import time
from functools import partial
from qiskit import QuantumCircuit, Aer, execute
from variationaltoolkit.utils import obj_from_statevector, precompute_obj
from variationaltoolkit.objectives import maxcut_obj
from variationaltoolkit.endianness import state_num2str


def local_pickleable_maxcut_obj(x, G=None):
    cut = 0
    for i, j in G.edges():
        if x[i] != x[j]:
            # the edge is cut
            cut -= 1
    return cut

class TestUtils(unittest.TestCase):

    def setUp(self):
        elist = [[0,1], [0,2], [0,3], [1,4], [1,5], [2,4], [2,5], [3,4], [3,5]]
        self.G=nx.OrderedGraph()
        self.G.add_edges_from(elist)
        self.obj = partial(local_pickleable_maxcut_obj, G=self.G)

    def test_obj_from_statevector(self):
        sv = np.zeros(2**6)
        sv[11] = 1
        self.assertTrue(np.isclose(obj_from_statevector(sv, self.obj), -5))

    def test_obj_from_statevector_complex(self):
        sv = np.zeros(2**6, dtype=complex)
        sv[11] = 1j
        self.assertTrue(np.isclose(obj_from_statevector(sv, self.obj), -5))

    def test_precompute_obj(self):
        G = nx.OrderedGraph()
        elist = [[0,1],[1,2],[1,3],[2,3]]
        G.add_edges_from(elist)
        N = G.number_of_nodes()
        w = nx.adjacency_matrix(G, nodelist=range(N))
        obj = partial(maxcut_obj, w=w)
        qc = QuantumCircuit(N,N)
        qc.x([0])
        backend = Aer.get_backend('statevector_simulator')
        sv = execute(qc, backend=backend).result().get_statevector()
        precomputed = precompute_obj(obj, N)
        self.assertEqual(len(precomputed[np.where(sv)]), 1)
        self.assertEqual(obj_from_statevector(sv, obj), precomputed[np.where(sv)][0])


if __name__ == '__main__':
    unittest.main()
