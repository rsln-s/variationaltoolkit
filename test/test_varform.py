import unittest
import numpy as np
import networkx as nx
import importlib.util
import sys
from itertools import product
from variationaltoolkit import VarForm
from variationaltoolkit.utils import mact, get_max_independent_set_operator
from qiskit.aqua.components.variational_forms import RYRZ
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
from qiskit.optimization.applications.ising.max_cut import get_operator as get_maxcut_operator

# a recipe for conditional import from https://docs.python.org/3/library/importlib.html#checking-if-a-module-can-be-imported
_mpsspec = importlib.util.find_spec('mpsbackend')

skip_mpsbackend = ('mpsbackend' not in sys.modules) and (_mpsspec is None)

class TestVarForm(unittest.TestCase):

    def setUp(self):
        self.varform_description = {'name':'RYRZ', 'num_qubits':5, 'depth':1}

    def test_import_ryrz(self):
        var_form = VarForm(varform_description=self.varform_description)
        self.assertIsInstance(var_form.var_form, RYRZ)
        
    def test_ryrz_qasm_simulator(self):
        var_form = VarForm(varform_description=self.varform_description)
        parameters = np.random.uniform(0, np.pi, var_form.num_parameters)
        execute_parameters={'shots':100}
        resstrs = var_form.run(parameters, 
                backend_description={'package':'qiskit', 'provider':'Aer', 'name':'qasm_simulator'},
                execute_parameters=execute_parameters)
        self.assertEqual(len(resstrs), execute_parameters['shots'])
        self.assertTrue(all(len(x) == self.varform_description['num_qubits'] for x in resstrs))

    def test_qaoa_maxcut(self):
        w = np.array([[0,1,1,0],[1,0,1,1],[1,1,0,1],[0,1,1,0]])
        C, offset = get_maxcut_operator(w)
        var_form = VarForm(varform_description={'name':'QAOA', 'p':2, 'cost_operator':C, 'num_qubits':4})
        parameters = np.random.uniform(0, np.pi, var_form.num_parameters)
        execute_parameters={'shots':100}
        resstrs = var_form.run(parameters, 
                backend_description={'package':'qiskit', 'provider':'Aer', 'name':'qasm_simulator'},
                execute_parameters=execute_parameters)
        self.assertEqual(len(resstrs), execute_parameters['shots'])
        self.assertTrue(all(len(x) == 4 for x in resstrs))


    def test_qaoa_mixer(self):
        w = np.array([[0,1,1,0],[1,0,1,1],[1,1,0,1],[0,1,1,0]])
        C, offset = get_maxcut_operator(w)
        # build initial state circuit
        initial_state_circuit = QuantumCircuit(4)
        initial_state_circuit.u2(0, np.pi, range(4))

        var_form_operator_mix = VarForm(varform_description={'name':'QAOA', 'p':2, 'cost_operator':C, 'num_qubits':4})
        var_form_circuit_mix = VarForm(varform_description={'name':'QAOA', 'p':2, 'cost_operator':C, 'num_qubits':4, 'use_mixer_circuit':True, 'initial_state_circuit':initial_state_circuit})

        self.assertEqual(var_form_operator_mix.num_parameters, var_form_circuit_mix.num_parameters)
        parameters = np.random.uniform(0, np.pi, var_form_operator_mix.num_parameters)
        sv_operator_mix = var_form_operator_mix.run(parameters, 
                backend_description={'package':'qiskit', 'provider':'Aer', 'name':'statevector_simulator'},
                execute_parameters={})
        sv_circuit_mix = var_form_circuit_mix.run(parameters, 
                backend_description={'package':'qiskit', 'provider':'Aer', 'name':'statevector_simulator'},
                execute_parameters={})
        # check that the two statevectors are equal up to global phase 
        phase_diff = sv_circuit_mix / sv_operator_mix
        self.assertTrue(np.allclose(phase_diff, np.full(phase_diff.shape, phase_diff[0])))


    def test_qaoa_pass_mixer(self):
        w = np.array([[0,1,1,0],[1,0,1,1],[1,1,0,1],[0,1,1,0]])
        C, offset = get_maxcut_operator(w)
        var_form_operator_mix = VarForm(varform_description={'name':'QAOA', 'p':2, 'cost_operator':C, 'num_qubits':4})
        # build initial state circuit
        initial_state_circuit = QuantumCircuit(4)
        initial_state_circuit.u2(0, np.pi, range(4))

        # build transverse field mixer circuit
        mixer_circuit = QuantumCircuit(4)
        beta = Parameter('beta')
        for q1 in range(4):
            mixer_circuit.h(q1)
            mixer_circuit.rz(2*beta, q1)
            mixer_circuit.h(q1)
        # pass it to variational form
        var_form_circuit_mix = VarForm(varform_description={'name':'QAOA', 'p':2, 'cost_operator':C, 'num_qubits':4, 'use_mixer_circuit':True, 'mixer_circuit':mixer_circuit, 'initial_state_circuit':initial_state_circuit})

        self.assertEqual(var_form_operator_mix.num_parameters, var_form_circuit_mix.num_parameters)
        parameters = np.random.uniform(0, np.pi, var_form_operator_mix.num_parameters)
        sv_operator_mix = var_form_operator_mix.run(parameters, 
                backend_description={'package':'qiskit', 'provider':'Aer', 'name':'statevector_simulator'},
                execute_parameters={})
        sv_circuit_mix = var_form_circuit_mix.run(parameters, 
                backend_description={'package':'qiskit', 'provider':'Aer', 'name':'statevector_simulator'},
                execute_parameters={})
        # check that the two statevectors are equal up to global phase 
        phase_diff = sv_circuit_mix / sv_operator_mix
        self.assertTrue(np.allclose(phase_diff, np.full(phase_diff.shape, phase_diff[0])))


    def test_qaoa_max_independent_set(self):
        elist = [[0,2],[0,4],[1,2],[1,4],[0,3],[1,3]]
        G = nx.OrderedGraph()
        G.add_edges_from(elist)
        vertex_num = G.number_of_nodes()
        w = nx.adjacency_matrix(G, nodelist=range(vertex_num))
        C, offset = get_max_independent_set_operator(vertex_num)
        # First, allocate registers
        qu = QuantumRegister(vertex_num)
        ancilla_for_multi_toffoli = QuantumRegister(vertex_num - 2)
        ancilla_for_rx = QuantumRegister(1)
        cu = ClassicalRegister(vertex_num)

        # Mixer circuit
        beta = Parameter('beta')
        mixer_circuit = QuantumCircuit(qu, ancilla_for_multi_toffoli, ancilla_for_rx, cu)
        for u in G.nodes():
            mixer_circuit.barrier()
            mact(mixer_circuit, list(qu[x] for x in G.neighbors(u)), ancilla_for_rx, ancilla_for_multi_toffoli)

            mixer_circuit.mcrx(2 * beta, ancilla_for_rx, qu[u])
            mixer_circuit.barrier()

            mact(mixer_circuit, list(qu[x] for x in G.neighbors(u)), ancilla_for_rx, ancilla_for_multi_toffoli)
        
        # Measurement circuit 
        measurement_circuit = QuantumCircuit(qu, ancilla_for_multi_toffoli, ancilla_for_rx, cu)
        measurement_circuit.measure(qu, cu)

        # pass it all to variational form
        var_form_circuit_mix = VarForm(varform_description={
            'name':'QAOA', 
            'p':2, 
            'cost_operator':C, 
            'num_qubits':vertex_num, 'use_mixer_circuit':True, 
            'mixer_circuit':mixer_circuit, 
            'measurement_circuit': measurement_circuit, 
            'qregs':[qu, ancilla_for_multi_toffoli, ancilla_for_rx, cu]})
        parameters = np.random.uniform(0, np.pi, var_form_circuit_mix.num_parameters)
        res = var_form_circuit_mix.run(parameters, 
                backend_description={'package':'qiskit', 'provider':'Aer', 'name':'qasm_simulator'},
                execute_parameters={'shots': 1000})
        # check that all sampled strings are valid solutions
        for x in res:
            for i, j in product(set(np.where(reversed(x))[0]), repeat=2):
                if i != j:
                    try:
                        self.assertTrue(not G.has_edge(i,j))
                    except AssertionError as e:
                        print(i,j,x)
                        raise e


    @unittest.skipIf(skip_mpsbackend, "mpsbackend not found")
    def test_ryrz_mpssimulator(self):
        var_form = VarForm(varform_description=self.varform_description)
        parameters = np.random.uniform(0, np.pi, var_form.num_parameters)
        execute_parameters={'shots':100}
        resstrs = var_form.run(parameters, 
                backend_description={'package':'mpsbackend'},
                execute_parameters=execute_parameters)
        self.assertEqual(len(resstrs), execute_parameters['shots'])
        self.assertTrue(all(len(x) == self.varform_description['num_qubits'] for x in resstrs))

# TODO:

# check that runs on qiskit.Aer mps simulator and returns correct number of resstrs
# check that submits a job to IBMQ correctly

if __name__ == '__main__':
    unittest.main()
