#!/usr/bin/env python

import numpy as np
import networkx as nx
from functools import partial
from qiskit.quantum_info import Pauli
from qiskit.aqua.operators import WeightedPauliOperator
from variationaltoolkit.objectivewrapper import ObjectiveWrapper
from variationaltoolkit.objectives import modularity_obj, bin_to_dec
from variationaltoolkit.utils import brute_force, obj_from_statevector, state_to_ampl_counts
from variationaltoolkit.endianness import get_adjusted_state



def get_modularity_4_operator(B, m):
    """Generate Hamiltonian for the modularity maximization with 4 communities.
    Args:
        B (numpy.ndarray) : modularity matrix.
        m (int)           : number of edges.
    Returns:
        WeightedPauliOperator: operator for the Hamiltonian
        float: a constant shift for the obj function.
    """
    num_nodes = B.shape[0]
    pauli_list = []
    shift = 0

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                x_p = np.zeros(num_nodes*2, dtype=np.bool)
                z_p = np.zeros(num_nodes*2, dtype=np.bool)
                z_p[2*i] = True
                z_p[2*j] = True
                pauli_list.append([-B[i, j] / (8 * m), Pauli(z_p, x_p)])

                x_p = np.zeros(num_nodes*2, dtype=np.bool)
                z_p = np.zeros(num_nodes*2, dtype=np.bool)
                z_p[2*i+1] = True
                z_p[2*j+1] = True
                pauli_list.append([-B[i, j] / (8 * m), Pauli(z_p, x_p)])

                x_p = np.zeros(num_nodes*2, dtype=np.bool)
                z_p = np.zeros(num_nodes*2, dtype=np.bool)
                z_p[2*i] = True
                z_p[2*j] = True
                z_p[2*i+1] = True
                z_p[2*j+1] = True
                pauli_list.append([-B[i, j] / (8 * m), Pauli(z_p, x_p)])

                shift -= B[i, j] / (8 * m)
            else:
                shift -= B[i, j] / (2 * m)

    return WeightedPauliOperator(paulis=pauli_list), shift


elist = [(0, 3), (0, 4), (1, 5), (1, 6), (2, 4), (2, 6), (3, 4), (3, 5), (4, 6)]
G = nx.OrderedGraph()
G.add_edges_from(elist)
node_list = list(range(G.number_of_nodes()))
nnodes = G.number_of_nodes()
nedges = len(elist)
assert(nedges == G.number_of_edges())
B = nx.modularity_matrix(G, nodelist=range(nnodes))

N = 2
obj_f = partial(modularity_obj, N=N, B=B, m=nedges)
opt_en, solution = brute_force(obj_f, nnodes*N)
print(f"True optimum: ", opt_en, solution)

p = 10
C, offset = get_modularity_4_operator(B, nedges)

obj_w = ObjectiveWrapper(obj_f, 
        varform_description={'name':'QAOA', 'p':p, 'cost_operator':C, 'num_qubits':G.number_of_nodes()*2}, 
        backend_description={'package':'qiskit', 'provider':'Aer', 'name':'statevector_simulator'}, 
        problem_description={'offset': offset, 'do_not_check_cost_operator':True},
        execute_parameters={}).get_obj()

theta = np.array([ 6.07687916,  9.04578039, 14.32475339, -1.83010038,  7.97646292,
        16.09278832,  3.90810118, 18.35957614, 20.29304497, 13.54441652,
         6.69979829,  0.98178805, -3.82011781, 10.4430878 , -7.31619394,
        14.06109113,  3.35586645, -2.39458044,  4.81429126, -7.40598448])

print(obj_w(theta))

print("---------------------------------------------------")

from qiskit.aqua.algorithms.adaptive.qaoa.var_form import QAOAVarForm
from qiskit import Aer, execute

varform = QAOAVarForm(C, p)
backend = Aer.get_backend('statevector_simulator')
def f(theta):
    qc = varform.construct_circuit(theta)
    sv = execute(qc, backend).result().get_statevector()
    # return the energy
    en = obj_from_statevector(sv, obj_f)
    return en

# Check that all is well
sv = np.zeros(2**(nnodes * N), dtype=complex)
sv[bin_to_dec(solution,nnodes * N)] = 1
print(obj_from_statevector(sv, obj_f))
adj_sv = get_adjusted_state(sv)
counts = state_to_ampl_counts(adj_sv)
assert(np.isclose(sum(np.abs(v)**2 for v in counts.values()), 1))
print(sum(obj_f(np.array([int(x) for x in k])) * (np.abs(v)**2) for k, v in counts.items()))


qc = varform.construct_circuit(theta)
sv = execute(qc, backend).result().get_statevector()
print(sv[:50])
print(obj_from_statevector(sv, obj_f))
