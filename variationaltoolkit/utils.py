import os
import numpy as np
import copy
import logging
from multiprocessing import Pool
from functools import partial
from qiskit import IBMQ
from qiskit import execute as qiskit_execute
from qiskit.providers.aer.backends.aerbackend import AerBackend
from qiskit.aqua.operators.op_converter import to_matrix_operator
from qiskit.quantum_info import Pauli
from qiskit.aqua.operators import WeightedPauliOperator

from .endianness import get_adjusted_state, state_num2str

logger = logging.getLogger(__name__)

def check_and_load_accounts():
    if IBMQ.active_account() is None:
        # try loading account
        IBMQ.load_account()
        if IBMQ.active_account() is None:
            # try grabbing token from environment
            logger.debug("Using token: {}".format(os.environ['QE_TOKEN']))
            IBMQ.enable_account(os.environ['QE_TOKEN'])

def execute_wrapper(experiments, backend, **kwargs):
    if isinstance(backend, AerBackend):
        return qiskit_execute(experiments, backend, **kwargs)
    else:
        from mpsbackend import MPSSimulator
        from mpsbackend import execute as mps_execute
        if isinstance(backend, MPSSimulator):
            return mps_execute(experiments, backend, **kwargs)
        else:
            raise ValueError(f"Unknown backend: {backend!r}")

def validate_objective(obj, problem_size):
    x = np.zeros(problem_size)
    try:
        obj(x)
    except Exception as e:
        print(f"objective function does not correctly accept lists of size {problem_size}, encountered {e}")
        raise e

def contains_and_raised(d, k):
    if d is None or k not in d:
        return False
    return bool(d[k])

def allclose_periodical(x, y, a, b, atol=1e-10):
    """
    Checks np.allclose(x,y), but assumes both x and y are periodical with respect to interval (a,b)
    """
    assert(len(x) == len(y))
    period = b-a
    x_p = np.remainder(x-a,period) # now in 0, b-a
    y_p = np.remainder(y-a,period)
    return all(np.isclose(x_p[i], y_p[i], atol=atol) or np.isclose(x_p[i], y_p[i]+period, atol=atol) or np.isclose(x_p[i], y_p[i]-period, atol=atol) for i in range(len(x_p)))

def state_to_ampl_counts(vec, eps=1e-15):
    """Converts a statevector to a dictionary
    of bitstrings and corresponding amplitudes
    """
    qubit_dims = np.log2(vec.shape[0])
    if qubit_dims % 1:
        raise ValueError("Input vector is not a valid statevector for qubits.")
    qubit_dims = int(qubit_dims)
    counts = {}
    str_format = '0{}b'.format(qubit_dims)
    for kk in range(vec.shape[0]):
        val = vec[kk]
        if val.real**2+val.imag**2 > eps:
            counts[format(kk, str_format)] = val
    return counts


def all_two_qubit_gates(qc):
    """
    Returns a list of the directions of all two-qubit gates (control, target)
    """
    res = []
    for g in qc:
        if len(g[1]) == 2:
            res.append((g[1][0].index, g[1][1].index))
    return res


def cost_operator_to_vec(C, offset=0):
    """Takes Qiskit WeightedPauliOperator
    representing the NxN cost Hamiltonian and converts
    it into a vector of length N of just the diagonal 
    elements. Verifies that C is real and diagonal.
    """
    C_mat = to_matrix_operator(C)
    m = C_mat.dense_matrix
    m_diag = np.zeros(m.shape[0])
    assert(m.shape[0] == m.shape[1])
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if i != j:
                assert(np.real(m[i][j]) == 0 and np.imag(m[i][j]) == 0)
            else:
                assert(np.imag(m[i][j]) == 0)
                m_diag[i] = np.real(m[i][j])
    return m_diag+offset


def precompute_obj(obj_f, nqubits):
    bitstrs = [state_num2str(k,nqubits)[::-1] for k in range(2**nqubits)] 
    return np.array([obj_f(np.array([int(y) for y in x])) for x in bitstrs])


def obj_from_statevector(sv, obj_f, precomputed=None):
    """Compute objective from Qiskit statevector
    For large number of qubits, this is slow. 
    To speed up for larger qubits, pass a vector of precomputed energies
    for QAOA, precomputed should be the same as the diagonal of the cost Hamiltonian
    """
    if precomputed is None:
        adj_sv = get_adjusted_state(sv)
        counts = state_to_ampl_counts(adj_sv)
        assert(np.isclose(sum(np.abs(v)**2 for v in counts.values()), 1))
        return sum(obj_f(np.array([int(x) for x in k])) * (np.abs(v)**2) for k, v in counts.items())
    else:
        return np.dot(precomputed, np.abs(sv)**2)


def check_cost_operator(C, obj_f, offset=0):
    """ Cost operator should be diagonal
    with the cost of state i as i-th element
    """
    m_diag = cost_operator_to_vec(C, offset=offset)
    m_diag = np.real(get_adjusted_state(m_diag))
    for k, v in state_to_ampl_counts(m_diag, eps=-1).items():
        x = np.array([int(_k) for _k in k])
        assert(np.isclose(obj_f(x), v))


def brute_force(obj_f, num_variables):
    best_cost_brute = 0
    for b in range(2**num_variables):
        x = [int(t) for t in reversed(list(bin(b)[2:].zfill(num_variables)))]
        try:
            cost = obj_f(x)
        except TypeError:
            cost = obj_f(np.array(x))
        if cost < best_cost_brute:
            best_cost_brute = cost
            xbest_brute = x
    return best_cost_brute, xbest_brute


def solution_density(obj_f, num_variables):
    solutions = []
    best_cost_brute = 0
    for b in range(2**num_variables):
        x = [int(t) for t in reversed(list(bin(b)[2:].zfill(num_variables)))]
        cost = obj_f(x)
        solutions.append(cost)
        if cost < best_cost_brute:
            best_cost_brute = cost
            xbest_brute = x
    assert(len(solutions) == 2**num_variables)
    noptimal = 0
    for cost in solutions:
        if np.isclose(cost, best_cost_brute):
            noptimal += 1
    return float(noptimal) / float(2**num_variables)


def set_log_level(level):
    """
    Sets logging level for everything in variationaltoolkit
    Args:
        level (logging.LEVEL): level to set 
    """
    root_logger = logging.getLogger('variationaltoolkit')
    root_logger.setLevel(level)
    for handler in root_logger.handlers:
        if isinstance(handler, type(logging.StreamHandler())):
            handler.setLevel(level)


def check_if_a_point_is_a_local_min(obj_f, x_min, eps=1e-2, f_min_precomputed=None):
    """
    Checks if points in local_mins array are indeed local minima
    by evaluating points eps away in each direction

    Args:
        obj_f (function): objective function 
        x_min (numpy.array or list): point to check
        eps (float): size of the step
        f_min_precomputed (float): function value at x_min to check (optional): 
                                   f_min_precomputed = obj_f(x_min)
    """
    f_min = obj_f(x_min)
    if f_min_precomputed is not None:
        assert(np.isclose(f_min, f_min_precomputed))
    for i, x in enumerate(x_min):
        x_eps = copy.deepcopy(x_min)
        x_eps[i] -= eps
        f_eps = obj_f(x_eps)
        if(f_eps <= f_min):
            return False
        x_eps[i] += 2*eps
        f_eps = obj_f(x_eps)
        if(f_eps <= f_min):
            return False
    return True


def mact(circuit, q_controls, q_target, ancilla):
    """
    Apply multiple anti-control Toffoli gate 

    Args:
        circuit (QuantumCircuit): The QuantumCircuit object to apply the mct gate on.
        q_controls (QuantumRegister or list(Qubit)): The list of control qubits
        q_target (Qubit): The target qubit
        q_ancilla (QuantumRegister or list(Qubit)): The list of ancillary qubits
    """
    circuit.x(q_controls)
    circuit.mct(q_controls, q_target[0], ancilla)
    circuit.x(q_controls)
    circuit.barrier()


def get_max_independent_set_operator(num_nodes):
    """
    Contructs the cost operator for max independent set
    1/2 \sum_i Z_i

    Args:
        num_nodes (int): Number of nodes
    """
    pauli_list = []
    for i in range(num_nodes):
        x_p = np.zeros(num_nodes, dtype=np.bool)
        z_p = np.zeros(num_nodes, dtype=np.bool)
        z_p[i] = True
        pauli_list.append([0.5, Pauli(z_p, x_p)])
    shift = -num_nodes/2
    return WeightedPauliOperator(paulis=pauli_list), shift


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