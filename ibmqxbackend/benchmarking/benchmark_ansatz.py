#!/usr/bin/env python

# To run:
#
# for i in $(seq 10 30); do ./benchmark_ansatz.py -q $i; done
#

import argparse
import timeit
import logging
import numpy as np
import networkx as nx
import csv
from pathlib import Path
from qiskit import Aer, execute
from ibmqxbackend.aqua.qaoa import QAOAVarForm
from ibmqxbackend.aqua.modularity_ising import get_modularity_qubitops
from qcommunity.utils.import_graph import generate_graph

parser = argparse.ArgumentParser()
parser.add_argument("-q", type=int, default=5, help="number of qubits / 2 (total number of qubits is 2*q)")
parser.add_argument("-threads", type=int, default=1, help="number of threads qiskit aer is using")
parser.add_argument(
    "--save",
    help="saves summarized results as a csv, with name as parameter. If csv exists, it appends to the end",
    type=str)
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)

p = 2
backend = Aer.get_backend("qasm_simulator")

G, _ = generate_graph("get_connected_caveman_graph", 2, args.q)
B = nx.modularity_matrix(G, weight='weight').A
qubitOp = get_modularity_qubitops(B)
var_form = QAOAVarForm(qubitOp, p)
parameters = np.random.uniform(-np.pi, np.pi, 2*p)
qc = var_form.construct_circuit(parameters)
qc.measure(qc.qregs[0], qc.cregs[0])
start_time = timeit.default_timer()
qobj = execute(qc, backend=backend, backend_options = {"max_parallel_threads" : args.threads})
res = qobj.result()
runtime = timeit.default_timer() - start_time
print("Finished in: {:.2f} sec on {} threads".format(runtime, args.threads))
header = ['nqubits', 'nthreads', 'runtime (sec)']
results = [args.q, args.threads, runtime]
if args.save:
    if Path(args.save).exists():
        write_header = False
    else:
        write_header = True

    with open(args.save, 'a') as csvfile:
        out = csv.writer(csvfile)
        if write_header:
            out.writerow(header)
        out.writerow(results)
