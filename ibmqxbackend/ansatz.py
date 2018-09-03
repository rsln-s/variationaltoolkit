#!/usr/bin/env python

from qiskit import register, available_backends
from qiskit import get_backend, compile
from ibmqxbackend.aqua.ryrz import VarFormRYRZ

class IBMQXVarForm(object):
    """
    Connection to IBM Quantum Experience that runs ansatz on IBM backend (be it simulator or physical device)

    By default uses Qconfig.py file in the root folder of ibmqxbackend module
    """

    def __init__(self, var_form='RYRZ', num_qubits=10, depth=3):
        try:
            import sys
            sys.path.append("../") # go to parent dir
            import Qconfig
        except Exception as e:
            print(e)
        
        APItoken=getattr(Qconfig, 'APItoken', None)
        url = Qconfig.config.get('url', None)
        hub = Qconfig.config.get('hub', None)
        group = Qconfig.config.get('group', None)
        project = Qconfig.config.get('project', None)
        try:
            register(APItoken, url, hub, group, project)
        except Exception as e:
            print(e)

        if var_form == 'RYRZ':
            self.var_form = VarFormRYRZ()
            self.var_form.init_args(num_qubits, depth, entanglement='linear')
            self.num_parameters = self.var_form._num_parameters

    def run(self, parameters, backend_name="local_qasm_simulator", return_all=False, samples=1000):
        res = {'backend_name':backend_name, 'parameters':parameters}
        qc = self.var_form.construct_circuit(parameters)

        # kinda hacky
        qc.measure(qc.get_qregs()['q'], qc.get_cregs()['c'])

        res['uncompiled_qasm'] = qc.qasm()
        backend = get_backend(backend_name)
        qobj = compile(qc, backend=backend, shots=samples)
        
        #res['compiled_qasm'] = qobj.experiments[0].header.compiled_circuit_qasm
        res['compiled_qasm'] = None 
        res['job'] = backend.run(qobj)
        res['result'] = res['job'].result()
        
        if return_all:
            return res 
        else:
            return None

