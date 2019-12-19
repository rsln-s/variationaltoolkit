#!/usr/bin/env python

from qiskit import IBMQ, Aer, execute, ClassicalRegister
from qiskit.providers.aer import noise
from qiskit.transpiler.passmanager import PassManager
from qiskit.aqua.algorithms.adaptive.qaoa.var_form import QAOAVarForm 
from qiskit.aqua.components.variational_forms.ryrz import RYRZ 
from ibmqxbackend.aqua.entangler_map import get_entangler_map_for_device
from ibmqxbackend.aqua.modularity_ising import get_modularity_qubitops
from ibmqxbackend.aqua.maxcut_ising import get_maxcut_qubitops
from ibmqxbackend.aqua.docplex_ising import get_general_ising_qubitops
from time import sleep
import os
import time
import datetime
import pytz
import logging
from pathlib import Path

# superhardcoded path to logfile
logfile_path = '/zfs/safrolab/users/rshaydu/quantum/data/ibmqxbackend_jobtimes_log.csv'

class IBMQXVarForm(object):
    """
    Connection to IBM Quantum Experience that runs ansatz on IBM backend (be it simulator or physical device)

    By default uses Qconfig.py file in the root folder of ibmqxbackend module
    """

    def __init__(self, problem_description, backend_params=None, depth=3, var_form='RYRZ', APItoken=None, target_backend_name=None):
        self.check_and_load_accounts()
        if backend_params is None:
            self.depth = depth
            self.var_form_name = var_form
        else:
            self.depth=backend_params['depth'] 
            self.var_form_name=backend_params['var_form'] 

        try:
            self.num_qubits = problem_description['num_qubits']
        except KeyError:
            self.num_qubits = problem_description['n_nodes']
        self.coupling_map = None
        self.noise_model = None
        self.basis_gates = None
        if self.var_form_name == 'RYRZ':
            self.target_backend_name = target_backend_name

            if self.target_backend_name is not None:
                # load noise models
                target_backend = IBMQ.get_backend(self.target_backend_name)
                properties = target_backend.properties()
                self.coupling_map = target_backend.configuration().coupling_map

                # Generate an Aer noise model for target_backend
                # self.noise_model = noise.device.basic_device_noise_model(properties) # unreliable
                # self.basis_gates = self.noise_model.basis_gates
            self.var_form = RYRZ(self.num_qubits, self.depth, entanglement='linear')
            self.shift = 0
        elif self.var_form_name == 'QAOA':
            if backend_params is not None and 'mixer_operator' in backend_params:
                mixerOp = backend_params['mixer_operator']
            else:
                mixerOp = None
            if problem_description['name'] == 'modularity':
                B = problem_description['B']
                if B is None:
                    raise ValueError("If using var_form == QAOA, have to specify B")
                qubitOp = get_modularity_qubitops(B)
                self.shift = 0
            elif problem_description['name'] == 'maxcut':
                A = problem_description['A']
                qubitOp, shift = get_maxcut_qubitops(A)
                self.shift = shift
            elif problem_description['name'] == 'ising':
                B_matrix = problem_description['B_matrix']
                B_bias = problem_description['B_bias']
                qubitOp, shift = get_general_ising_qubitops(B_matrix, B_bias)
                self.shift = shift
            else:
                raise ValueError("Unsupported problem: {}".format(problem_description['name']))
            self.var_form = QAOAVarForm(qubitOp, self.depth, mixer_operator=mixerOp)
        else:
            raise ValueError("Incorrect var_form {}".format(self.var_form_name))
        self.num_parameters = self.var_form.num_parameters
        logging.info("Initialized IBMQXVarForm {} with num_qubits={}, depth={}".format(self.var_form_name, self.num_qubits, self.depth))

    def check_and_load_accounts(self):
        if IBMQ.active_account() is None:
            # try loading account
            IBMQ.load_account()
            if IBMQ.active_account() is None:
                # try grabbing token from environment
                logging.debug("Using token: {}".format(os.environ['QE_TOKEN']))
                IBMQ.enable_account(os.environ['QE_TOKEN'])

    def run(self, parameters, backend_name="qasm_simulator", return_all=False, samples=1000, seed=42, nattempts=25):
        if backend_name is None or "simulator" in backend_name:
            backend = Aer.get_backend("qasm_simulator")
        else:
            self.check_and_load_accounts()
            backend = IBMQ.get_backend(backend_name)
        for attempt in range(0,nattempts):
            try:
                logging.debug("Using backend {}".format(backend_name))
                res = {'backend_name':backend_name, 'parameters':parameters}
                qc = self.var_form.construct_circuit(parameters)

                # kinda hacky
                if not qc.cregs:
                    c = ClassicalRegister(self.num_qubits, name='c')
                    qc.add_register(c)

                qc.measure(qc.qregs[0], qc.cregs[0])

                if backend_name is None or "simulator" in backend_name:
                    qobj = execute(qc, backend=backend, 
                            shots=samples, 
                            coupling_map=self.coupling_map, 
                            seed_simulator=seed,
                            pass_manager=PassManager(), # turn off all optimization for simulator
                            noise_model=None,
                            basis_gates=None)
                    res['result'] = qobj.result()
                else:
                    # quantum backend
                    start = time.time()
                    start_time_utc = pytz.utc.localize(datetime.datetime.utcnow())
                    qobj = execute(qc, backend=backend, 
                            shots=samples)
                    res['result'] = qobj.result()
                    end = time.time()
                    end_time_utc = pytz.utc.localize(datetime.datetime.utcnow())
                    timezone = "America/Chicago"
                    start_time_cst = start_time_utc.astimezone(pytz.timezone(timezone))
                    end_time_cst = end_time_utc.astimezone(pytz.timezone(timezone))
                    runtime_seconds = end - start
                    if Path(logfile_path).exists():
                        with open(logfile_path, 'a') as f:
                            line = ','.join([str(x) for x in [qobj.job_id(), 
                                                              start_time_cst.strftime("%Y-%m-%d %H:%M:%S"),
                                                              end_time_cst.strftime("%Y-%m-%d %H:%M:%S"),
                                                              timezone,
                                                              runtime_seconds,
                                                              backend_name]])
                            f.write(line+'\n')
                
                if return_all:
                    return res 
                else:
                    resstrs = []
                    for k, v in res['result'].get_counts().items():
                        resstrs.extend([tuple(int(x) for x in k)]*v)
                    return resstrs
            except (KeyboardInterrupt, SystemExit):
                raise
            except () as e:
                sleep(attempt * 10)
                if attempt < nattempts - 1:
                    logging.warning("While using IBM Q backend, encountered {}. Trying again...".format(e))
                    # retry
                    continue
                else:
                    # propagate the error
                    raise e
            break
