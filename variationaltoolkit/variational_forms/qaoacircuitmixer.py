"""
QAOA with circuit mixer instead of mixer operator.
Adapted from qiskit.aqua.algorithms.adaptive.qaoa.var_form.QAOAVarForm (copyright notice copied below)
"""

# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Global X phases and parameterized problem hamiltonian."""

from functools import reduce

import numpy as np
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.quantum_info import Pauli
from qiskit.circuit import Parameter
from qiskit.aqua.operators import (OperatorBase, X, I, H, Zero, CircuitStateFn, EvolutionFactory, LegacyBaseOperator)
from qiskit.aqua.operators import WeightedPauliOperator, op_converter
from qiskit.aqua.components.variational_forms import VariationalForm
from qiskit.aqua.components.initial_states import InitialState
# pylint: disable=invalid-name


class QAOACircuitMixer(VariationalForm):
    """Global X phases and parameterized problem hamiltonian."""

    def __init__(self, cost_operator, p, initial_state_circuit=None, mixer_circuit=None, measurement_circuit=None, qregs=None):
        """
        Constructor, following the QAOA paper https://arxiv.org/abs/1411.4028

        Args:
            cost_operator (WeightedPauliOperator): The operator representing the cost of
                                                   the optimization problem,
                                                   denoted as U(B, gamma) in the original paper.
            p (int): The integer parameter p, which determines the depth of the circuit,
                     as specified in the original paper.
            initial_state_circuit (QuantumCircuit, optional): Circuit that prepares the initial state.
            mixer_circuit (QuantumCircuit, optional): An optional custom mixer operator
                                                              to use instead of
                                                              the global X-rotations,
                                                              denoted as U(B, beta)
                                                              in the original paper.
                                                      Mixer circuit should be a parameterized
                                                      QuantumCircuit with parameter beta
                                                      See default for example
            measurement_circuit (QuantumCircuit, optional): An optional circuit with measurements (useful if your mixer has ancillas in it)
            qregs (list of QuantumRegister) : Registers (also vis-a-vis ancillas)
                                              qregs[0] MUST BE the register with working qubits (i.e. the ones that are measured in the end)
        Raises:
            TypeError: invalid input
        """
        super().__init__()
        self._cost_operator = cost_operator
        self._num_qubits = cost_operator.num_qubits
        self._p = p
        self._initial_state_circuit = initial_state_circuit
        self._num_parameters = 2 * p
        self._bounds = [(0, np.pi)] * p + [(0, 2 * np.pi)] * p
        self._preferred_init_points = [0] * p * 2

        # prepare the mixer operator
        if isinstance(self._initial_state_circuit, LegacyBaseOperator):
            self._initial_state_circuit = self._initial_state_circuit.to_opflow()
        if mixer_circuit is None:
            num_qubits = self._num_qubits
            mixer_terms = [(I ^ left) ^ X ^ (I ^ (num_qubits - left - 1)) for left in range(num_qubits)]
            self._mixer_circuit = sum(mixer_terms)
        elif isinstance(mixer_circuit, LegacyBaseOperator):
            self._mixer_circuit = mixer_circuit.to_opflow()
        else:
            if not isinstance(mixer_circuit, QuantumCircuit):
                raise TypeError('The mixer should be a qiskit.QuantumCircuit '
                                + 'object, found {} instead'.format(type(mixer_circuit)))
            self._mixer_circuit = mixer_circuit
       # if len(self._mixer_circuit.parameters) != 1:
        #    raise ValueError(f"Mixer circuit should have exactly one parameter (beta), received {self._mixer_circuit.parameters}")
        self.support_parameterized_circuit = True
        self._measurement_circuit = measurement_circuit
        self.qregs = qregs

    def construct_circuit(self, parameters):
        """ construct circuit """
        angles = parameters
        if not len(angles) == self.num_parameters:
            raise ValueError('Incorrect number of angles: expecting {}, but {} given.'.format(
                self.num_parameters, len(angles)
            ))

        # initialize circuit, possibly based on given register/initial state
        if self.qregs is None:
            self.qregs = [QuantumRegister(self._num_qubits, name='q')]
            
        circuit = (H ^ self._num_qubits)

        if self._initial_state_circuit is not None:
            init_state = CircuitStateFn(self._initial_state_circuit.construct_circuit('circuit'))
        else:
            init_state = Zero
        
        for idx in range(self._p):
            beta, gamma = angles[idx], angles[idx + self._p]
            circuit = (self._cost_operator * beta).exp_i().compose(circuit)
            circuit = (self._mixer_circuit * gamma).exp_i().compose(circuit)

        evolution = EvolutionFactory.build(self._cost_operator)
        circuit = evolution.convert(circuit)
        if self._measurement_circuit is not None:
            circuit += self._measurement_circuit
        return circuit.to_circuit()

    @property
    def setting(self):
        """ returns setting """
        ret = "Variational Form: {}\n".format(self.__class__.__name__)
        params = ""
        for key, value in self.__dict__.items():
            if key != "_configuration" and key[0] == "_":
                params += "-- {}: {}\n".format(key[1:], value)
        ret += "{}".format(params)
        return ret
