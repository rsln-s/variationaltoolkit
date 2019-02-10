#!/usr/bin/env python

from qiskit import IBMQ, Aer
from qiskit import compile, QISKitError
from ibmqxbackend.aqua.ryrz import VarFormRYRZ
from time import sleep
import os
import logging

class IBMQXVarForm(object):
    """
    Connection to IBM Quantum Experience that runs ansatz on IBM backend (be it simulator or physical device)

    By default uses Qconfig.py file in the root folder of ibmqxbackend module
    """

    def __init__(self, problem_description, depth=3, var_form='RYRZ', APItoken=None):
        if len(IBMQ.active_accounts()) <= 1:
            # try just loading
            IBMQ.load_accounts()
            # if that didn't work, resort to grabbing tokens
            if len(IBMQ.active_accounts()) <= 1: 
                # try grabbing token from environment
                logging.debug("Using token: {}".format(os.environ['QE_TOKEN']))
                IBMQ.enable_account(os.environ['QE_TOKEN'], os.environ['QE_URL'])
        num_qubits = problem_description['num_qubits']
        if var_form == 'RYRZ':
            self.var_form = VarFormRYRZ()
            self.var_form.init_args(num_qubits, depth, entanglement='linear')
            self.num_parameters = self.var_form._num_parameters
        else:
            raise ValueError("Incorrect var_form {}".format(var_form))
        logging.debug("Initialized IBMQXVarForm {} with num_qubits={}, depth={}".format(var_form, num_qubits, depth))

    def run(self, parameters, backend_name="qasm_simulator", return_all=False, samples=1000, seed=42, nattempts=25):
        if backend_name is None or "simulator" in backend_name:
            backend = Aer.get_backend("qasm_simulator")
        else:
            backend = IBMQ.get_backend(backend_name)
        for attempt in range(0,nattempts):
            try:
                logging.info("Using backend {}".format(backend_name))
                res = {'backend_name':backend_name, 'parameters':parameters}
                qc = self.var_form.construct_circuit(parameters)

                # kinda hacky
                qc.measure(qc.qregs[0], qc.cregs[0])

                res['uncompiled_qasm'] = qc.qasm()
                qobj = compile(qc, backend=backend, shots=samples, seed=seed)

                res['job'] = backend.run(qobj)
                res['result'] = res['job'].result()
                
                if return_all:
                    return res 
                else:
                    resstrs = []
                    for k, v in res['result'].get_counts().items():
                        resstrs.extend([[int(x) for x in k]]*v)
                    return resstrs
            except (QISKitError, KeyError) as e:
                sleep(attempt * 10)
                if attempt < nattempts - 1:
                    logging.warning("While using IBM Q backend, encountered {}. Trying again...".format(e))
                    # retry
                    continue
                else:
                    # propagate the error
                    raise e
            break
