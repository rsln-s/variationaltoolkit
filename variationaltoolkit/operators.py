import numpy as np

from qiskit.quantum_info import Pauli
from qiskit.aqua.operators import WeightedPauliOperator


def get_mis_w_penalty_operator(G):
    """Generate Hamiltonian for the max-cut problem of a graph.
    Args:
        weight_matrix (numpy.ndarray) : adjacency matrix.
    Returns:
        WeightedPauliOperator: operator for the Hamiltonian
        float: a constant shift for the obj function.
    """
    num_nodes = G.number_of_nodes()
    pauli_list = []
    shift = 0
    # first,W =  \sim_i 1/2(I-Z_i)
    for i in range(num_nodes):
        x_p = np.zeros(num_nodes, dtype=np.bool)
        z_p = np.zeros(num_nodes, dtype=np.bool)
        z_p[i] = True
        pauli_list.append([0.5, Pauli(z_p, x_p)])
        shift -= 0.5
    # second, C_IS = \sum b_ib_j
    for i, j in G.edges():
        # 1/4 Z_i
        x_p = np.zeros(num_nodes, dtype=np.bool)
        z_p = np.zeros(num_nodes, dtype=np.bool)
        z_p[i] = True
        pauli_list.append([-0.25, Pauli(z_p, x_p)])
        # 1/4 Z_j
        x_p = np.zeros(num_nodes, dtype=np.bool)
        z_p = np.zeros(num_nodes, dtype=np.bool)
        z_p[j] = True
        pauli_list.append([-0.25, Pauli(z_p, x_p)])
        # 1/4 Z_jZ_i
        x_p = np.zeros(num_nodes, dtype=np.bool)
        z_p = np.zeros(num_nodes, dtype=np.bool)
        z_p[j] = True
        z_p[i] = True
        pauli_list.append([0.25, Pauli(z_p, x_p)])        
        shift += 0.25
    return WeightedPauliOperator(paulis=pauli_list), shift