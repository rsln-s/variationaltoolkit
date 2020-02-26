import os
import numpy as np
import logging
from multiprocessing import Pool
from functools import partial
from qiskit import IBMQ
from qiskit import execute as qiskit_execute
from qiskit.providers.aer.backends.aerbackend import AerBackend
from qiskit.aqua.operators.op_converter import to_matrix_operator

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
    

def pickleable_wrapper_with_parameter_return_and_conversion_to_int(x, obj_f=None):
    return x, obj_f(np.array([int(y) for y in x]))


def precompute_obj(obj_f, nqubits, nprocesses=1):
    obj_f_w = partial(pickleable_wrapper_with_parameter_return_and_conversion_to_int, obj_f=obj_f)

    with Pool(nprocesses) as p:
        bitstrs = [state_num2str(k,nqubits) for k in range(2**nqubits)]
        return dict(p.map(obj_f_w, bitstrs))


def obj_from_statevector(sv, obj_f, precomputed=None):
    """Compute objective from Qiskit statevector
    For large number of qubits, this is very slow. 
    """
    adj_sv = get_adjusted_state(sv)
    counts = state_to_ampl_counts(adj_sv)
    assert(np.isclose(sum(np.abs(v)**2 for v in counts.values()), 1))
    if precomputed is None:
        return sum(obj_f(np.array([int(x) for x in k])) * (np.abs(v)**2) for k, v in counts.items())
    else:
        return sum(precomputed[k] * (np.abs(v)**2) for k, v in counts.items())


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
        cost = obj_f(x)
        if cost < best_cost_brute:
            best_cost_brute = cost
            xbest_brute = x
    return best_cost_brute, xbest_brute
