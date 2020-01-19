import unittest
import numpy as np
import networkx as nx
from variationaltoolkit.objectives import modularity_obj

class TestObjectives(unittest.TestCase):

    def test_modularity_obj(self):
        w = np.array([[0,1,1,0,0,0],[1,0,1,0,0,0],[1,1,0,1,0,0],[0,0,1,0,1,1],[0,0,0,1,0,1],[0,0,0,1,1,0]])
        G = nx.from_numpy_matrix(w)
        for node in G.nodes():
            G.nodes[node]['volume'] = G.degree[node]
        for u, v in G.edges():
            G[u][v]['weight'] = 1
        node_list = list(G.nodes())
        x = np.array([0,0,0,1,1,1])
        N = 1
        y = np.array([0,0,0,1,1,0,1,1,1,1,1,1])
        M = 2
        self.assertTrue(modularity_obj(x, N, G, node_list) + 10/28 < 1e-5)
        self.assertTrue(modularity_obj(y, M, G, node_list) + 9/98 < 1e-5)


if __name__ == '__main__':
    unittest.main()
