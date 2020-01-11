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
