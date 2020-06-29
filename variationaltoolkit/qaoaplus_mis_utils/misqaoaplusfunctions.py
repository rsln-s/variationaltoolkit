import numpy as np
import networkx as nx
import time
from functools import partial
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.optimization.ising.max_cut import get_operator as get_maxcut_operator
from variationaltoolkit.objectives import maxcut_obj, modularity_obj
from variationaltoolkit import VariationalQuantumOptimizerSequential
from variationaltoolkit.operators import get_mis_w_penalty_operator

"""
The function that perform one run of the QAOAplus

Parameters
----------
p : QAOA+'s p value
G : the input graph
initial_state_string: a string of initial state, default is ''
print_out_res : a boolean that allows the user to print out the res for debug

Returns
-------
counts: the dictionary that shows the final quantum state outputs
"""
def qaoaplus_one_run(p, G, initial_state_string='', print_out_res=False):
    import logging
    from variationaltoolkit.utils import set_log_level
    # set_log_level(logging.INFO)
    
    vertex_num = G.number_of_nodes()

    # set up the objective function for MIS QAOA+
    def mis_w_penalty(x, G):
        obj = -sum(x)
        for i, j in G.edges():
            obj += (x[i] * x[j])
        return obj
    obj = partial(mis_w_penalty, G=G)

    # set up cost Hamiltonian
    # call function from Qiskit
    # can also set up the circuit version manually
    C, offset = get_mis_w_penalty_operator(G)
    # print('Cost Hamiltonian is')
    # print(C.print_details())

    # manually set up the mixer circuit
    mixer_circuit = QuantumCircuit(vertex_num)
    beta = Parameter('beta')
    for q1 in range(vertex_num):
        mixer_circuit.h(q1)
        mixer_circuit.rz(2*beta, q1)
        mixer_circuit.h(q1)
    # print('mixer_circuit is')
    # print(mixer_circuit)
    # print()

    # manually set up the initial state circuit
    assert isinstance(initial_state_string, str), "need to pass the initial state as a string"
    if initial_state_string != '':
        assert len(initial_state_string) == vertex_num, "string length need to equal the number of nodes"
    initial_state = initial_state_string
    initial_state_circuit = QuantumCircuit(vertex_num)
    for i in range(len(initial_state)):
        current_state = initial_state[i]
        # Qiskit is doing in reverse order. For the first number in the initial_state, it means the last qubit
        actual_i = len(initial_state) - 1 - i
        if current_state == '1':
            initial_state_circuit.x(actual_i)
    # print(initial_state_circuit)
    

    varopt = VariationalQuantumOptimizerSequential(
        obj,
        'COBYLA',
        initial_point=None,
        optimizer_parameters={'maxiter':1000},
        backend_description={'package':'qiskit', 'provider':'Aer', 'name':'qasm_simulator'},
        problem_description={'offset': offset, 'do_not_check_cost_operator':True},
        varform_description={
            'name':'QAOA',
            'p':p,
            'cost_operator':C,
            'num_qubits':vertex_num, 'use_mixer_qaoaplus':True,
            'mixer_circuit':mixer_circuit,
            'initial_state_circuit':initial_state_circuit,
        },
        execute_parameters={'shots': 5000}
        )

    res = varopt.optimize()
    if print_out_res == True:
        print(res)
    
    optimum, counts = varopt.get_optimal_solution(return_counts=True)
    # print(counts)
    print(f"Found optimal solution: {optimum}")
    return counts


"""
The function that compute the average energy after pruning, also record total number of feasible solutions

Parameters
----------
p : QAOA+'s p value
G : the input graph
initial_state_string: a string of initial state, default is ''

Returns
-------
energy_feasible: the average energy after pruning
"""
def qaoaplus_compute_energy_avg_with_pruning(p, G, initial_state_string=''):
    def mis_obj(x, G):
        obj = -sum(x)
        return obj
    
    # define the function that check if a string solution is an independent set
    def is_independent_set(x, G):
        for i, j in G.edges():
            if x[i]*x[j] == 1:
                 return False
        return True
    # is_feasible = partial(is_independent_set, G=G)
    
    energy_feasible = 0
    feasible_count = 0
    total_count = 0
    counts = qaoaplus_one_run(p, G, initial_state_string)
    for k,v in counts.items():
        k_string_array = np.array([int(i) for i in k])
        # invert the string order
        k_string_array = k_string_array[::-1]
        if is_independent_set(k_string_array, G):
            energy_feasible += mis_obj(k_string_array, G) * v
            feasible_count += v
        total_count += v
    energy_feasible /= feasible_count
    
    print("the energy of the solution is " + str(energy_feasible))
    print('feasible solution percetage ratio is')
    print(str(feasible_count / total_count * 100) + '%')
    print()
    
    return energy_feasible


"""
The function that gives the energy average and mininum for many runs of the MIS QAOA+

Parameters
----------
p : QAOA's p value
num_experiment_iter: number of experimental iteration
G: the input graph
initial_state_string: a string of initial state, default is ''

Returns
-------
energy_avg: the computed energy average
energy_min: the computed energy min
"""
def qaoaplus_compute_energy_average_min(p, num_experiment_iter, G, initial_state_string=''):
    energy_avg = 0
    energy_min = 0
    for i in range(num_experiment_iter):
        current_energy = qaoaplus_compute_energy_avg_with_pruning(p, G, initial_state_string)
        energy_avg += current_energy
        if current_energy < energy_min:
            energy_min = current_energy
    energy_avg /= num_experiment_iter
    print('the average energy is : ' + str(energy_avg))
    print('the mininum energy is : ' + str(energy_min))
    return energy_avg, energy_min
