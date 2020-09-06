import unittest
import numpy as np
import networkx as nx
import time
from functools import partial
from qiskit import QuantumCircuit, Aer, execute
from qiskit.optimization.ising.max_cut import get_operator as get_maxcut_operator
from variationaltoolkit.utils import obj_from_statevector, precompute_obj, cost_operator_to_vec, solution_density, get_max_independent_set_operator, check_cost_operator, get_modularity_4_operator
from variationaltoolkit.objectives import maxcut_obj, modularity_obj
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

    def test_precompute_obj_cost_ham(self):
        w = nx.adjacency_matrix(self.G, nodelist=range(self.G.number_of_nodes()))
        C, offset = get_maxcut_operator(w)
        cost_diag = cost_operator_to_vec(C, offset)
        precomputed = precompute_obj(self.obj, self.G.number_of_nodes())
        self.assertTrue(np.allclose(cost_diag, precomputed))

    def test_solution_density(self):
        G = nx.generators.classic.complete_graph(8)
        obj_f = partial(local_pickleable_maxcut_obj, G=G)
        density = solution_density(obj_f, G.number_of_nodes())
        self.assertEqual(density, 0.2734375)


    def test_get_max_independent_set_operator(self):
        n = 4
        def obj(x):
            return -sum(x)
        C, offset = get_max_independent_set_operator(n)
        check_cost_operator(C, obj, offset=offset)
        
    def test_get_modularity_4_operator(self):
        G = nx.fast_gnp_random_graph(6, 0.4, seed = 0, directed=False)
        elist = [e for e in G.edges]
        G = nx.OrderedGraph()
        G.add_edges_from(elist)
        node_list = list(range(G.number_of_nodes()))
        B = nx.modularity_matrix(G, nodelist = node_list)
        m = G.number_of_edges()
        C, offset = get_modularity_4_operator(B, m)
        obj_f = partial(modularity_obj, N=2, B = B,m = m)
        check_cost_operator(C, obj_f, offset = offset)



if __name__ == '__main__':
    unittest.main()
