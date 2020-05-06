import unittest
import numpy as np
import networkx as nx
from variationaltoolkit.objectives import modularity_obj

class TestObjectives(unittest.TestCase):

    def test_modularity_obj(self):
        w = np.array([[0,1,1,0,0,0],[1,0,1,0,0,0],[1,1,0,1,0,0],[0,0,1,0,1,1],[0,0,0,1,0,1],[0,0,0,1,1,0]])
        G = nx.from_numpy_matrix(w)
        B = nx.modularity_matrix(G, nodelist = list(range(6)))
        m = G.number_of_edges()
        
        x = np.array([0,0,0,1,1,1])
        N = 1
        y = np.array([0,0,0,1,1,0,1,1,1,1,1,1])
        M = 2
        
        self.assertTrue(abs(modularity_obj(x, N, B, m) + 10/28) < 1e-5)
        self.assertTrue(abs(modularity_obj(y, M, B, m) + 9/98) < 1e-5)


if __name__ == '__main__':
    unittest.main()
