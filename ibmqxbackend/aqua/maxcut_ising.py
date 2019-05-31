"""

Generates cost Hamiltonian for the maxcut problem

from qiskit_aqua/translators/ising/maxcut.py

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
from qiskit_aqua import Operator

def get_maxcut_qubitops(weight_matrix):
    """Generate Hamiltonian for the maximum stableset in a graph.
    
    Args:
        weight_matrix (numpy.ndarray) : adjacency matrix.
    
    Returns:
        operator.Operator, float: operator for the Hamiltonian and a
        constant shift for the obj function.
    
    """
    num_nodes = weight_matrix.shape[0]
    pauli_list = []
    shift = 0
    for i in range(num_nodes):
        for j in range(i):
            if (weight_matrix[i,j] != 0):
                wp = np.zeros(num_nodes)
                vp = np.zeros(num_nodes)
                vp[i] = 1
                vp[j] = 1
                pauli_list.append([0.5 * weight_matrix[i, j], Pauli(vp, wp)])
                shift -= 0.5 * weight_matrix[i, j]
    return Operator(paulis=pauli_list), shift
