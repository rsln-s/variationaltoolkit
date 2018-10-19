#!/usr/bin/env python

# A little example script for Yuri

# To run:
#
# ./yuri.py -q 4 
#

from qiskit import IBMQ, Aer 
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-q", type=int, default=5, help="number of qubits")
parser.add_argument("-backend", type=str, default="qasm_simulator", help="backend name")
args = parser.parse_args()

# Import my wrapper
from ibmqxbackend.ansatz import IBMQXVarForm

# Initialize hardware-efficient ansatz of desired depth
var_form = IBMQXVarForm(num_qubits=args.q, depth=2)

# Generate random angles (QAOA variational parameters)
parameters = np.random.uniform(-np.pi, np.pi, var_form.num_parameters)

# All available backends
print("Backends: ", IBMQ.backends(), Aer.backends())

backend_name = args.backend

print("running on {}...".format(backend_name))

res = var_form.run(parameters, backend_name=backend_name, return_all=False, samples=10)

print(res)
