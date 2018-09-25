#!/usr/bin/env python

from qiskit import register, available_backends, registered_providers
from qiskit import get_backend, compile, QISKitError
from ibmqxbackend.aqua.ryrz import VarFormRYRZ
from time import sleep
import os
import logging

class IBMQXVarForm(object):
    """
    Connection to IBM Quantum Experience that runs ansatz on IBM backend (be it simulator or physical device)

    By default uses Qconfig.py file in the root folder of ibmqxbackend module
    """

    def __init__(self, num_qubits=10, depth=3, var_form='RYRZ', APItoken=None):
        providers = registered_providers()
        if len(providers) <= 1:
            # if didn't register yet
            if APItoken is None:
                # try grabbing token from environment
                logging.info("Using token: {}".format(os.environ['QE_TOKEN']))
                register(os.environ['QE_TOKEN'])
            else:
                logging.info("Using token: {}".format(APItoken))
                register(APItoken)

        if var_form == 'RYRZ':
            self.var_form = VarFormRYRZ()
            self.var_form.init_args(num_qubits, depth, entanglement='linear')
            self.num_parameters = self.var_form._num_parameters
        else:
            raise ValueError("Incorrect var_form {}".format(var_form))
        logging.info("Initialized IBMQXVarForm {} with num_qubits={}, depth={}".format(var_form, num_qubits, depth))

    def run(self, parameters, backend_name="local_qasm_simulator", return_all=False, samples=1000, seed=42, nattempts=25):
        if backend_name is None:
            backend_name = "local_qasm_simulator"
        for attempt in range(0,nattempts):
            try:
                logging.info("Using backend {}".format(backend_name))
                res = {'backend_name':backend_name, 'parameters':parameters}
                qc = self.var_form.construct_circuit(parameters)

                # kinda hacky
                qc.measure(qc.get_qregs()['q'], qc.get_cregs()['c'])

                res['uncompiled_qasm'] = qc.qasm()
                backend = get_backend(backend_name)
                qobj = compile(qc, backend=backend, shots=samples, seed=seed)
                
                res['compiled_qasm'] = qobj['circuits'][0]['compiled_circuit_qasm'] 
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
                    print("While using IBM Q backend, encountered {}. Trying again...".format(e))
                    # retry
                    continue
                else:
                    # propagate the error
                    raise e
            break
