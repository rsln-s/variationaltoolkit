import numpy as np
import logging
import copy
from operator import itemgetter
import variationaltoolkit.optimizers as vt_optimizers
import qiskit.aqua.components.optimizers as qiskit_optimizers
from .objectivewrapper import ObjectiveWrapper
from .objectivewrappersmooth import ObjectiveWrapperSmooth
from .utils import state_to_ampl_counts, check_cost_operator, get_adjusted_state, contains_and_raised

logger = logging.getLogger(__name__)

class VariationalQuantumOptimizerQuantumFlow:
    def __init__(self, obj, optimizer_name, initial_point=None, variable_bounds=None, optimizer_parameters=None, objective_parameters=None, varform_description=None, backend_description=None, problem_description=None, execute_parameters=None):
        """Constuctor.
        
        Args:
            obj (function) : takes a list of 0,1 and returns objective function value for that vector
            optimizer_name        (str) : optimizer name. Should correspond to some tensorflow optimizer, e.g. Adam (TODO: figure this out!)
            initial_point    (np.array) : initial point for the optimizer
            variable_bounds (list[(float, float)]) : list of variable
                                            bounds, given as pairs (lower, upper). None means
                                            unbounded.
            optimizer_parameters (dict) : Parameters for the variational parameter optimizer.
                                          Transarently passed to qiskit aqua optimizers.
                                          See docs for corresponding optimizer.
            varform_description (dict)  : See varform.py
            problem_description (dict)  : 'offset': difference between the energies of the cost Hamiltonian and true energies of obj (same as in qiskit)
                                          'do_not_check_cost_operator': does not run check_cost_operator on cost operator, which is slow for large number of qubits. Use with caution! 
                                          'smooth_schedule' : tries fitting a smooth schedule instead of optimizing directily (see objectivewrappersmooth.py)
            backend_description (dict)  : See varform.py
            execute_parameters (dict)   : See objectivewrapper.py
            objective_parameters (dict) : See objectivewrapper.py 
        """
        self.obj = obj
        if varform_description['name'] == 'QAOA':
            if 'offset' in problem_description:
                offset=problem_description['offset']
            else:
                offset=0
            if not contains_and_raised(problem_description, 'do_not_check_cost_operator'):
                if varform_description['cost_operator'].num_qubits >= 10:
                    logger.warning('check_cost_operator requires building full density matrix, prohibitive for high number of qubits \n Recommended to set: problem_description[\'do_not_check_cost_operator\']=True')
                check_cost_operator(varform_description['cost_operator'], obj, offset=offset)
        if variable_bounds is not None:
            self.variable_bounds = variable_bounds
        else:
            self.variable_bounds = [(None, None)] * self.obj_w.num_parameters 
        if initial_point is not None:
            self.initial_point=initial_point
        else:
            lb = [(l if l is not None else -2 * np.pi) for (l, u) in self.variable_bounds]
            ub = [(u if u is not None else 2 * np.pi) for (l, u) in self.variable_bounds]
            self.initial_point = np.random.uniform(lb,ub, self.obj_w.num_parameters)
        self.varform_description = varform_description
        self.problem_description = problem_description
        self.execute_parameters = execute_parameters
        self.objective_parameters = objective_parameters
        self.backend_description = backend_description
        self.res = {}
        # TODO add additional initialization as needed


    def optimize(self):
        """Minimize the objective
        """

        # TODO: must populate the following fields in the result dict and return it  

        self.res['num_optimizer_evals'] = None 
        self.res['min_val'] = None
        self.res['opt_params'] = None

        return self.res
