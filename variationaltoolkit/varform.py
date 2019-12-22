import importlib.util
import sys

# a recipe for conditional import from https://docs.python.org/3/library/importlib.html#checking-if-a-module-can-be-imported
name = 'mpsbackend'
spec = importlib.util.find_spec(name)
if name in sys.modules:
    raise RuntimeError(f"{name!r} already in sys.modules. This means that someone imported {name!r} before, braking the encapsulation. This should not happen.")
elif spec is not None:
    # If you chose to perform the actual import ...
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    print(f"Using {name!r}")
else:
    print(f"Can't find the {name!r} module, continuing without it")

from qiskit import IBMQ, Aer, execute, ClassicalRegister
import qiskit.aqua.components.variational_forms as qiskit_variational_forms

class VarForm:
    """Variational Form wrapper"""

    def __init__(self, varform_description=None, backend_description=None, problem_description=None):
        """Constructor.

        Args:
            varform_description (dict) : Specifies the variational form.
                                         Must specify name
                                         For hardware-efficient variational forms, 
                                         must match the __init__ of desired variational form.
                                         For QAOA, must specify p
            backend_description (dict) : Specifies backend parameters.
                                         TBA
                                         For qiskit, must specify TBA 
            problem_description (dict) : Specifies the problem (maxcut, modularity, ising). 
                                         Optional for hardware-efficient variational forms.
                                         Must have field 'name'.
        """

        if varform_description is None or backend_description is None:
            raise ValueError(f"varform_description = {varform_description!r} and backend_description = {backend_description!r} are required")
    
        if varform_description['name'] == 'QAOA':
            raise NotImplementedError('QAOA not supported yet')
        
        varform_parameters = {k : v for k,v in varform_description.items() if k != 'name'}

        self.var_form = getattr(qiskit_variational_forms, varform_description['name'])(**varform_parameters)

        self.varform_description = varform_description
        self.backend_description = backend_description
        self.problem_description = problem_description
