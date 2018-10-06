#!/usr/bin/env python

# To run:
#
# for i in $(seq 10 30); do ./benchmark_ansatz.py -q $i; done
#

# Import the Qiskit SDK
from qiskit import IBMQ, Aer 
import time
import numpy as np
from difflib import ndiff
import argparse
import logging
import sys
import pickle
#from qiskit.backends.jobstatus import JobStatus, JOB_FINAL_STATES
from ibmqxbackend.ansatz import IBMQXVarForm

parser = argparse.ArgumentParser()
parser.add_argument("-q", type=int, default=5, help="number of qubits")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)

var_form = IBMQXVarForm(num_qubits=args.q, depth=0)

parameters = np.random.uniform(-np.pi, np.pi, var_form.num_parameters)

# See a list of available local simulators
print("Backends: ", IBMQ.backends(), Aer.backends())
#backend_name = "ibmq_5_tenerife"
#backend_name = "ibmq_16_melbourne"
backend_name = "qasm_simulator"
#backend_name = "local_qasm_simulator_py"

print("running on {}...".format(backend_name))

#res = var_form.run(parameters, backend_name=backend_name, return_all=True)
import timeit
start_time = timeit.default_timer()
res = var_form.run(parameters, backend_name=backend_name)
print("Finished in: ", timeit.default_timer() - start_time)

#print(res['uncompiled_qasm'])
#print('------------------------------')
#print(res['compiled_qasm'])

#print(res)

if False:
    # set to True to see what changed in compilation
    print("Changed in compilation:")
    diff = ndiff(res['uncompiled_qasm'].splitlines(keepends=True), res['compiled_qasm'].splitlines(keepends=True))
    print(''.join(diff), end="")

#outname = 'stuff/rueschlikon.p'
#pickle.dump(res, open(outname, "wb"))

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


#print(res)
#import pdb
#pdb.set_trace()

#print(res['result'])
# Show the results
#counts = result.get_counts()
#print(counts)
