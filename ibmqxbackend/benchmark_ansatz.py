#!/usr/bin/env python

# To run:
#
# for i in $(seq 10 30); do ./benchmark_ansatz.py -q $i; done
#

# Import the Qiskit SDK
from qiskit import available_backends
import time
import numpy as np
from difflib import ndiff
import argparse
#from qiskit.backends.jobstatus import JobStatus, JOB_FINAL_STATES
from ibmqxbackend.ansatz import IBMQXVarForm

parser = argparse.ArgumentParser()
parser.add_argument("-q", type=int, default=10, help="number of qubits")
args = parser.parse_args()

var_form = IBMQXVarForm(num_qubits=args.q, depth=3)

parameters = np.random.uniform(-np.pi, np.pi, var_form.num_parameters)

# See a list of available local simulators
print("Backends: ", available_backends(compact=False))

#backend_name = "ibmq_5_tenerife"
#backend_name = "ibmq_16_rueschlikon"
backend_name = "local_qasm_simulator_cpp"
#backend_name = "local_qasm_simulator_py"

if False:
    # set to True to see what changed in compilation
    print("Changed in compilation:")
    diff = ndiff(uncompiled_qasm.splitlines(keepends=True), compiled_qasm.splitlines(keepends=True))
    print(''.join(diff), end="")
    sys.exit(0)

print("running on {}...".format(backend_name))

import timeit

start_time = timeit.default_timer()
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

res = var_form.run(parameters, backend_name=backend_name)

print(res)
#import pdb
#pdb.set_trace()

elapsed = timeit.default_timer() - start_time
print("For {} qubits finished in {}".format(args.q,elapsed))
#print(res['result'])
# Show the results
#counts = result.get_counts()
#print(counts)
