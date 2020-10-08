import numpy as np
import networkx as nx
import sys
from itertools import product
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
import matplotlib.pyplot as plt
from variationaltoolkit import VarForm
from variationaltoolkit.utils import mact, get_max_independent_set_operator
from variationaltoolkit import VariationalQuantumOptimizerSequential

"""
The function that perform one run of the MIS QAOA using Stuart's ansatz

Parameters
----------
p : QAOA's p value
G : input graph
initial_state_string: a string of initial state, default is ''
initial_state_variation: a string indicating a specific state, default is 'all_zero'
print_out_res : a boolean that allows the user to print out the res for debug

Returns
-------
counts: the dictionary that shows the final quantum state outputs
"""

def stuart_one_run(p, G, initial_state_string='', initial_state_variation='all_zero', print_out_res=False, return_optimum=False, optimizer='COBYLA'):
    vertex_num = G.number_of_nodes()
    
    def obj(x):
        return -sum(x)

    # Set level to INFO to print optimization progress
    import logging
    from variationaltoolkit.utils import set_log_level
    # set_log_level(logging.INFO)

    # Generate cost Hamiltonian
    C, offset = get_max_independent_set_operator(vertex_num)


    # Mixer circuit
    beta = Parameter('beta')
    # First, allocate registers
    qu = QuantumRegister(vertex_num)
    ancilla_for_multi_toffoli = QuantumRegister(vertex_num - 2)
    ancilla_for_rx = QuantumRegister(1)
    cu = ClassicalRegister(vertex_num)
        
    mixer_circuit = QuantumCircuit(qu, ancilla_for_multi_toffoli, ancilla_for_rx, cu)
    
    for u in G.nodes():
        neighbor_list = list(G.neighbors(u))
        if not neighbor_list:
            mixer_circuit.rx(2 * beta, qu[u])
        else:
            mixer_circuit.barrier()
            mact(mixer_circuit, list(qu[x] for x in G.neighbors(u)), ancilla_for_rx, ancilla_for_multi_toffoli)

            mixer_circuit.mcrx(2 * beta, ancilla_for_rx, qu[u])
            mixer_circuit.barrier()

            mact(mixer_circuit, list(qu[x] for x in G.neighbors(u)), ancilla_for_rx, ancilla_for_multi_toffoli)

    # Measurement circuit
    measurement_circuit = QuantumCircuit(qu, ancilla_for_multi_toffoli, ancilla_for_rx, cu)
    measurement_circuit.measure(qu, cu)


    # manually set up the initial state circuit
    initial_state_circuit = QuantumCircuit(vertex_num)
    
    assert isinstance(initial_state_string, str), "need to pass the initial state as a string"
    if initial_state_string != '':
        assert len(initial_state_string) == vertex_num, "string length need to equal the number of nodes"
    for i in range(len(initial_state_string)):
        current_state = initial_state_string[i]
        # Qiskit is doing in reverse order. For the first number in the initial_state, it means the last qubit
        actual_i = len(initial_state_string) - 1 - i
        if current_state == '1':
            initial_state_circuit.x(actual_i)
    # print(initial_state_string)


    # pass it all to VariationalQuantumOptimizer
    varopt = VariationalQuantumOptimizerSequential(
        obj,
        optimizer,
        initial_point=None,
        optimizer_parameters={'maxiter':1000},
        backend_description={'package':'qiskit', 'provider':'Aer', 'name':'qasm_simulator'},
        problem_description={'offset': offset, 'do_not_check_cost_operator':True},
        varform_description={
            'name':'QAOA',
            'p':p,
            'cost_operator':C,
            'num_qubits':vertex_num, 'use_mixer_circuit':True,
            'mixer_circuit':mixer_circuit,
            'measurement_circuit': measurement_circuit,
            'initial_state_circuit':initial_state_circuit,
            'qregs':[qu, ancilla_for_multi_toffoli, ancilla_for_rx, cu],
        },
        execute_parameters={'shots': 5000}
        )

    res = varopt.optimize()
    if print_out_res == True:
        print(res)
    optimum, counts = varopt.get_optimal_solution(return_counts=True)
    print(f"Found optimal solution: {optimum}")
    if return_optimum == True:
        return optimum, counts
    else:
        return counts, res