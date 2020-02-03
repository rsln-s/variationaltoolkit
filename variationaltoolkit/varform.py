import importlib.util
import sys
import numpy as np

# a recipe for conditional import from https://docs.python.org/3/library/importlib.html#checking-if-a-module-can-be-imported
_spec = importlib.util.find_spec('mpsbackend')
if 'mpsbackend' in sys.modules:
    raise RuntimeError("mpsbackend already in sys.modules. This means that someone imported mpsbackend before, braking the encapsulation. This should not happen.")
elif _spec is not None:
    # If you chose to perform the actual import ...
    _module = importlib.util.module_from_spec(_spec)
    sys.modules['mpsbackend'] = _module
    _spec.loader.exec_module(_module)
    from mpsbackend import MPSSimulator
    print("Using mpsbackend")
else:
    print(f"Can't find the mpsbackend module, continuing without it")

import qiskit
import qiskit.aqua.components.variational_forms as qiskit_variational_forms
from qiskit.aqua.algorithms.adaptive.qaoa.var_form import QAOAVarForm
from .utils import execute_wrapper, check_and_load_accounts

class VarForm:
    """Variational Form wrapper"""

    def __init__(self, varform_description=None, problem_description=None):
        """Constructor.

        Args:
            varform_description (dict) : Specifies the variational form.
                                         Must specify name
                                         For hardware-efficient variational forms, 
                                         must match the __init__ of desired variational form.
                                         For QAOA, must specify p, cost_operator
                                         optional: mixer_operator
            problem_description (dict) : Specifies the problem (maxcut, modularity, ising). 
                                         Optional for hardware-efficient variational forms.
                                         Must have field 'name'.
        """

        if varform_description is None:
            raise ValueError(f"varform_description is required")
    

        self.num_qubits = varform_description['num_qubits']
        if varform_description['name'] == 'QAOA':
            varform_parameters = {k : v for k,v in varform_description.items() if k != 'name' and k != 'num_qubits'}
            self.var_form = QAOAVarForm(**varform_parameters)
        else:
            varform_parameters = {k : v for k,v in varform_description.items() if k != 'name'}
            self.var_form = getattr(qiskit_variational_forms, varform_description['name'])(**varform_parameters)
        self.num_parameters = self.var_form._num_parameters

        self.varform_description = varform_description
        self.problem_description = problem_description

    def run(self, parameters, backend_description=None, execute_parameters=None):
        """Runs the variational form

        Args:
            parameters (np.array) : variational parameters to pass to the form
            backend_description (dict) : Specifies backend parameters.
                                         TBA
                                         For qiskit, must specify TBA 
            execute_parameters (dict)  : Parameters passed to execute function
        """
        if backend_description is None:
            raise ValueError(f"backend_description is required")

        if backend_description['package'] == 'qiskit':
            if backend_description['provider'] == 'Aer':
                provider = qiskit.Aer
            else:
                check_and_load_accounts()
                provider = qiskit.IBMQ.get_provider(backend_description['provider'])
            backend = provider.get_backend(backend_description['name'])
        elif backend_description['package'] == 'mpsbackend':
            backend = MPSSimulator()

        circuit = self.var_form.construct_circuit(parameters)

        if backend_description['package'] == 'qiskit' and 'statevector' not in backend_description['name']:
            if not circuit.cregs:
                c = qiskit.ClassicalRegister(self.num_qubits, name='c')
                circuit.add_register(c)

            circuit.measure(circuit.qregs[0], circuit.cregs[0])

        job = execute_wrapper(circuit, backend, **execute_parameters)
        result = job.result()

        if backend_description['package'] == 'qiskit' and 'statevector' in backend_description['name']:
            return result.get_statevector()
        else:
            if hasattr(result, 'get_resstrs'):
                return result.get_resstrs()
            else:
                resstrs = []
                for k, v in result.get_counts().items():
                    for _ in range(v):
                        resstrs.append(np.array([int(x) for x in k]))
                return resstrs
