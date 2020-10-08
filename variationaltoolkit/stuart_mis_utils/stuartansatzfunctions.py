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
initial_state_variation: a string indicating a specific state, default is 'all_zero', can set to 'w_state' here
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
            
    if initial_state_variation == 'w_state':
        for i in range(1, vertex_num):
            initial_state_circuit.cx(0, i)
            initial_state_circuit.x(i)
            initial_state_circuit.ry(np.arcsin(1 / ((vertex_num + 1 - i) ** (1/2))), 0)
            initial_state_circuit.cx(i, 0)
            initial_state_circuit.ry(-np.arcsin(1 / ((vertex_num + 1 - i) ** (1/2))), 0)
            initial_state_circuit.x(i)
            initial_state_circuit.cx(0, i)

            initial_state_circuit.x(0) # This is not covered in the paper
            initial_state_circuit.x(i) # This is not covered in the paper
            if i > 1:
                initial_state_circuit.swap(0, i) # This is not covered in the paper

        initial_state_circuit.x(0) # flip back the first qubit
    
    # print(initial_state_circuit)


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


"""
The function that calculate the energy for one run of the MIS QAOA using Stuart's ansatz

Parameters
----------
p : QAOA's p value
G : input graph
initial_state_string: a string of initial state, default is ''

Returns
-------
energy_feasible: the computed energy for the feasible/all solution
"""
def stuart_compute_energy_avg(p, G, initial_state_string=''):
    counts = stuart_one_run(p, G, initial_state_string)
    def mis_obj(x, G):
            obj = -sum(x)
            return obj

    from qiskit.tools.visualization import plot_histogram
    def is_independent_set(x, G):
        for i, j in G.edges():
            if x[i]*x[j] == 1:
                return False
        return True
    # is_feasible = partial(is_independent_set, G=G)
    energy_feasible = 0
    feasible_count = 0
    total_count = 0
    
    for k,v in counts.items():
        k_string_array = np.array([int(i) for i in k])
        # invert the string order
        k_string_array = k_string_array[::-1]
        if is_independent_set(k_string_array, G):
            energy_feasible += mis_obj(k_string_array, G) * v
            feasible_count += v
        total_count += v
    
    if feasible_count != total_count:
        raise ValueError(f"Encountered {total_count - feasible_count} samples that are not independent sets")
        
    energy_feasible /= feasible_count
    return energy_feasible


"""
Function that gives the energy minimum for many runs of the MIS Stuart's ansatz for the same graph
Can select with or without prunning

Parameters
----------
p : QAOA's p value
num_experiment_iter: number of experimental iteration
G: the input graph
initial_state_string: a string of initial state, default is 'all_zero'
"""
def stuart_compute_min(p, num_experiment_iter, G, initial_state_variation='all_zero'):
    energy_min = 0
    for i in range(num_experiment_iter):
        print ("experiment=",i)
        counts, res = stuart_one_run(p, G, initial_state_variation=initial_state_variation, print_out_res=False)
        current_energy = res['min_val']
        if current_energy < energy_min:
            energy_min = current_energy
    print('the mininum energy is : ' + str(energy_min))
    return energy_min


"""
Function that print out an array filled with energy output from many runs of stuart_comute_min() with different graphs
Can select with or without prunning

Parameters
----------
n: number of nodes (for Erdős–Rényi graphs)
d: degree of nodes (for Erdős–Rényi graphs)
p : QAOA's p value
num_experiment_iter: number of experimental iteration
graph_seed_begin: the first seed to generate the Erdős–Rényi graphs
graph_seed_end: the last seed to generate the Erdős–Rényi graphs
initial_state_string: a string of initial state, default is 'all_zero'
"""
def stuart_min_energy_array(n, d, p, num_experiment_iter, graph_seed_begin, graph_seed_end, initial_state_variation='all_zero'):
    A=[]
    for seed in range(graph_seed_begin, graph_seed_end + 1):
        print("seed=", seed)
        G = nx.gnp_random_graph(n, d/n, seed=seed)
        min_energy = stuart_compute_min(p, num_experiment_iter, G, initial_state_variation='all_hadamard')
        print("min_energy=", min_energy)
        A.append(min_energy)
    print('the energy array is: ')
    print(A)
    print('the average energy over many graphs is: ')
    print(sum(A)/len(A))
    # print(0.969* n/d)


"""
(archived method in case needed, try using stuart_compute_min())
The function that gives the energy average and mininum for one run of the MIS QAOA using Stuart's ansatz

Parameters
----------
p : QAOA's p value
num_experiment_iter: number of experimental iteration
G: input graph
initial_state_string: a string of initial state, default is ''

Returns
-------
energy_avg: the computed energy average
energy_min: the computed energy min
"""
def stuart_compute_energy_average_min(p, num_experiment_iter, G, initial_state_string='', print_min_average = False):
    energy_avg = 0
    energy_min = 0
    for i in range(num_experiment_iter):
        current_energy = stuart_compute_energy_avg(p, G, initial_state_string)
        print()
        energy_avg += current_energy
        if current_energy < energy_min:
            energy_min = current_energy
    energy_avg /= num_experiment_iter
    if print_min_average == True:
        print('the average energy is : ' + str(energy_avg))
        print('the mininum energy is : ' + str(energy_min))
    return energy_avg, energy_min
