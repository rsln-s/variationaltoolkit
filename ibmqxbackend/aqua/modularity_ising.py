"""

Generates cost Hamiltonian for the modularity maximization problem

"""
import numpy as np
from qiskit.quantum_info import Pauli
from qiskit.aqua import Operator

def get_modularity_qubitops(B):
    num_nodes = B.shape[0]
    pauli_list = []
    for i in range(num_nodes):
        for j in range(i):
            if (B[i,j] != 0):
                wp = np.zeros(num_nodes)
                vp = np.zeros(num_nodes)
                vp[i] = 1
                vp[j] = 1
                pauli_list.append([2.0 * B[i, j], Pauli(vp, wp)])
    return Operator(paulis=pauli_list)
