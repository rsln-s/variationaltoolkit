import unittest
import numpy as np
import networkx as nx
from variationaltoolkit.utils import obj_from_statevector

class TestUtils(unittest.TestCase):

    def setUp(self):
        elist = [[0,1], [0,2], [0,3], [1,4], [1,5], [2,4], [2,5], [3,4], [3,5]]
        G=nx.OrderedGraph()
        G.add_edges_from(elist)
        def maxcut_obj(x):
            cut = 0
            for i, j in G.edges():
                if x[i] != x[j]:
                    # the edge is cut
                    cut -= 1
            return cut
        self.obj = maxcut_obj

    def test_obj_from_statevector(self):
        sv = np.zeros(2**6)
        sv[11] = 1
        self.assertTrue(np.isclose(obj_from_statevector(sv, self.obj), -5))

    def test_obj_from_statevector_complex(self):
        sv = np.zeros(2**6, dtype=complex)
        sv[11] = 1j
        self.assertTrue(np.isclose(obj_from_statevector(sv, self.obj), -5))




if __name__ == '__main__':
    unittest.main()
