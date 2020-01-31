import os
import numpy as np
from qiskit import IBMQ
from qiskit import execute as qiskit_execute
from qiskit.providers.aer.backends.aerbackend import AerBackend

def check_and_load_accounts():
    if IBMQ.active_account() is None:
        # try loading account
        IBMQ.load_account()
        if IBMQ.active_account() is None:
            # try grabbing token from environment
            logging.debug("Using token: {}".format(os.environ['QE_TOKEN']))
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
