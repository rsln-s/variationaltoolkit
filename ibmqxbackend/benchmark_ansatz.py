#!/usr/bin/env python

# To run:
#
# for i in $(seq 10 30); do ./benchmark_ansatz.py -q $i; done
#

# Import the Qiskit SDK
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import available_backends, execute, get_backend, compile
import time
import numpy as np
from difflib import ndiff
import argparse
#from qiskit.backends.jobstatus import JobStatus, JOB_FINAL_STATES

parser = argparse.ArgumentParser()
parser.add_argument("-q", type=int, default=10, help="number of qubits")
args = parser.parse_args()

try:
    import sys
    sys.path.append("../") # go to parent dir
    import Qconfig
except Exception as e:
    print(e)
    
from qiskit import register, available_backends

#set api
APItoken=getattr(Qconfig, 'APItoken', None)
url = Qconfig.config.get('url', None)
hub = Qconfig.config.get('hub', None)
group = Qconfig.config.get('group', None)
project = Qconfig.config.get('project', None)
try:
    register(APItoken, url, hub, group, project)
except Exception as e:
    print(e)

#-------------------------------------------------- 
# TODO: insert variational form ansatz
# ./qiskit_aqua/utils/variational_forms/ryrz.py or smth
#-------------------------------------------------- 

from ibmqxbackend.aqua.ryrz import VarFormRYRZ

var_form = VarFormRYRZ()
num_qubits = args.q
depth = 3
var_form.init_args(num_qubits, depth, entanglement='linear')
parameters = np.random.uniform(-np.pi, np.pi, var_form._num_parameters)

qc = var_form.construct_circuit(parameters)

# relies on hacked VarFormRYRZ
qc.measure(qc.get_qregs()['q'], qc.get_cregs()['c'])

uncompiled_qasm = qc.qasm()

# See a list of available local simulators
#print("Backends: ", available_backends(compact=False))

#backend_name = "ibmq_5_tenerife"
#backend_name = "ibmq_16_rueschlikon"
backend_name = "local_qasm_simulator_cpp"

# Compile and run the Quantum circuit on a simulator backend
my_backend = get_backend(backend_name)
qobj = compile(qc, backend=my_backend)

#compiled_qasm = qobj.experiments[0].header.compiled_circuit_qasm

if False:
    # set to True to see what changed in compilation
    print("Changed in compilation:")
    diff = ndiff(uncompiled_qasm.splitlines(keepends=True), compiled_qasm.splitlines(keepends=True))
    print(''.join(diff), end="")
    sys.exit(0)

print("running on {}...".format(backend_name))

import timeit

start_time = timeit.default_timer()
job = my_backend.run(qobj)
#lapse = 0
#interval = 1
#while True:
#    print('Status @ {} seconds'.format(interval * lapse))
#    print(job.status())
#    if job.status() in JOB_FINAL_STATES:
#        break
#    time.sleep(interval)
#    lapse += 1
#
#print(job.status)

result = job.result()
elapsed = timeit.default_timer() - start_time
print("For {} qubits finished in {}".format(args.q,elapsed))
print(result)
# Show the results
#counts = result.get_counts()
#print(counts)
