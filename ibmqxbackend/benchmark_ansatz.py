#!/usr/bin/env python

# To run:
#
# for i in $(seq 10 30); do ./benchmark_ansatz.py -q $i; done
#

import argparse
import timeit
import logging
import numpy as np
from qcommunity.optimization.obj import get_obj_val

parser = argparse.ArgumentParser()
parser.add_argument("-q", type=int, default=5, help="number of qubits / 2 (total number of qubits is 2*q)")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)

p = 2
obj_val, num_parameters = get_obj_val("get_connected_caveman_graph", 2, args.q, seed=1, obj_params='ndarray', backend='IBMQX', backend_params={'backend_device': 'qasm_simulator', 'depth': p, 'var_form':'QAOA'}, sign=-1)
y = np.random.uniform(-np.pi, np.pi, num_parameters)
start_time = timeit.default_timer()
print(obj_val(y))
print("Finished in: {:.2f} sec".format(timeit.default_timer() - start_time))
