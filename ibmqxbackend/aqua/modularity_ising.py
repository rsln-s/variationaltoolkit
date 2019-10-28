"""

Generates cost Hamiltonian for the modularity maximization problem

"""
import numpy as np
from qiskit.quantum_info import Pauli
from qiskit.aqua.operators import WeightedPauliOperator

def get_modularity_qubitops(B):
    """Generate cost Hamiltonian for the modularity maximization problem

    Args:
        B (numpy.ndarray) : modularity matrix.
    Returns:
        WeightedPauliOperator: operator for the Hamiltonian
    """
    num_nodes = B.shape[0]
    pauli_list = []
    for i in range(num_nodes):
        for j in range(i):
            if (B[i,j] != 0):
                wp = np.zeros(num_nodes, dtype=np.bool)
                vp = np.zeros(num_nodes, dtype=np.bool)
                vp[i] = True
                vp[j] = True
                pauli_list.append([2.0 * B[i, j], Pauli(vp, wp)])
    return WeightedPauliOperator(paulis=pauli_list)
