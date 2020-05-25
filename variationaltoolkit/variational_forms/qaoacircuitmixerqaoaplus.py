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

from qiskit.aqua.operators import WeightedPauliOperator, op_converter
from qiskit.aqua.components.variational_forms import VariationalForm

# pylint: disable=invalid-name


class QAOACircuitMixerQAOAPlus(VariationalForm):
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
            initial_state_circuit (QuantumCircuit): Circuit that prepares the initial state.
            mixer_circuit (QuantumCircuit): A custom mixer operator
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
        cost_operator = op_converter.to_weighted_pauli_operator(cost_operator)
        self._cost_operator = cost_operator
        self._num_qubits = cost_operator.num_qubits
        self._p = p
        self._initial_state_circuit = initial_state_circuit
        self._num_parameters = 1 + 2 * p
        # self._bounds = [(0, np.pi)] * p + [(0, 2 * np.pi)] * p
        self._preferred_init_points = [0] * (1 + p * 2)

        # prepare the mixer operator
        v = np.zeros(self._cost_operator.num_qubits)
        ws = np.eye(self._cost_operator.num_qubits)
        if mixer_circuit is None:
            # default mixer is transverse field
            self._mixer_circuit = QuantumCircuit(self._num_qubits)
            beta = Parameter('beta')
            for q1 in range(self._num_qubits):
                self._mixer_circuit.h(q1)
                self._mixer_circuit.rz(2*beta, q1)
                self._mixer_circuit.h(q1)
        else:
            if not isinstance(mixer_circuit, QuantumCircuit):
                raise TypeError('The mixer should be a qiskit.QuantumCircuit '
                                + 'object, found {} instead'.format(type(mixer_circuit)))
            self._mixer_circuit = mixer_circuit
        if len(self._mixer_circuit.parameters) != 1:
            raise ValueError(f"Mixer circuit should have exactly one parameter (beta), received {self._mixer_circuit.parameters}")
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
        circuit = QuantumCircuit(*self.qregs)
        if self._initial_state_circuit is not None:
            circuit += self._initial_state_circuit
            
        # add the first 0.5p part
        theta = angles[0]
        theta_parameter = self._mixer_circuit.parameters.pop()
        circuit += self._mixer_circuit.bind_parameters({theta_parameter: theta})

        # add the remaining p cost evolution and p mixer evolution
        for idx in range(self._p):
            beta, gamma = angles[idx + 1], angles[idx + self._p + 1]
            circuit += self._cost_operator.evolve(
                evo_time=gamma, num_time_slices=1, quantum_registers=self.qregs[0]
            )
            beta_parameter = self._mixer_circuit.parameters.pop() # checked in constructor that there's only one parameter
            circuit += self._mixer_circuit.bind_parameters({beta_parameter: beta})
        if self._measurement_circuit is not None:
            circuit += self._measurement_circuit
        return circuit

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
