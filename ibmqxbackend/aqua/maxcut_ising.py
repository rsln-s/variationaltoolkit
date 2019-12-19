"""

Generates cost Hamiltonian for the maxcut problem

from qiskit.aqua/translators/ising/maxcut.py

# Copyright 2018 IBM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""


import numpy as np
from qiskit.quantum_info import Pauli
from qiskit.aqua.operators import WeightedPauliOperator

def get_maxcut_qubitops(weight_matrix):
    """Generate Hamiltonian for the max-cut problem of a graph.
    Args:
        weight_matrix (numpy.ndarray) : adjacency matrix.
    Returns:
        WeightedPauliOperator: operator for the Hamiltonian
        float: a constant shift for the obj function.
    """
    num_nodes = weight_matrix.shape[0]
    pauli_list = []
    shift = 0
    for i in range(num_nodes):
        for j in range(i):
            if weight_matrix[i, j] != 0:
                x_p = np.zeros(num_nodes, dtype=np.bool)
                z_p = np.zeros(num_nodes, dtype=np.bool)
                z_p[i] = True
                z_p[j] = True
                pauli_list.append([0.5 * weight_matrix[i, j], Pauli(z_p, x_p)])
                shift -= 0.5 * weight_matrix[i, j]
    return WeightedPauliOperator(paulis=pauli_list), shift
