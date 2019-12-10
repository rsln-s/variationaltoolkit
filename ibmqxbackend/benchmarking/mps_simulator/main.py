#!/usr/bin/env python

# benchmarks mps QasmSimulator

import argparse
import timeit
import numpy as np
import csv
from pathlib import Path
from qiskit import Aer, execute, ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit.aqua.components.variational_forms import RYRZ


parser = argparse.ArgumentParser()
parser.add_argument("-q", type=int, default=5, help="number of qubits / 2 (total number of qubits is 2*q)")
parser.add_argument("-d", type=int, default=1, help="depth (number of layers)")
parser.add_argument(
    "--save",
    help="saves summarized results as a csv, with name as parameter. If csv exists, it appends to the end",
    type=str)
args = parser.parse_args()

var_form = RYRZ(args.q, depth=args.d, entanglement='linear', entanglement_gate='cx')
parameters = np.random.uniform(0, np.pi, var_form._num_parameters)
qc = var_form.construct_circuit(parameters)
if not qc.cregs:
    c = ClassicalRegister(var_form._num_qubits, name='c')
    qc.add_register(c)
qc.measure(qc.qregs[0], qc.cregs[0])

# Select the QasmSimulator from the Aer provider
simulator = Aer.get_backend('qasm_simulator')

# Define the simulation method
backend_opts_mps = {"method":"matrix_product_state"}

start_time = timeit.default_timer()
result = execute(qc, simulator, backend_options=backend_opts_mps).result()
runtime = timeit.default_timer() - start_time
print("Finished in: {:.2f} sec with {} qubits, {} depth".format(runtime, args.q, args.d))


header = ['nqubits', 'depth', 'runtime (sec)']
results = [args.q, args.d, runtime]
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
