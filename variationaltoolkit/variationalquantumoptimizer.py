import numpy as np
import logging
import copy
from operator import itemgetter
from abc import ABC, abstractmethod
import variationaltoolkit.optimizers as vt_optimizers
import qiskit.aqua.components.optimizers as qiskit_optimizers
from .objectivewrapper import ObjectiveWrapper
from .objectivewrappersmooth import ObjectiveWrapperSmooth
from .utils import state_to_ampl_counts, check_cost_operator, get_adjusted_state, contains_and_raised
from collections import Counter

logger = logging.getLogger(__name__)

class VariationalQuantumOptimizer(ABC):
    def __init__(self, obj, optimizer_name, initial_point=None, variable_bounds=None, optimizer_parameters=None, objective_parameters=None, varform_description=None, backend_description=None, problem_description=None, execute_parameters=None):
        """Constuctor.
        
        Args:
            obj (function) : takes a list of 0,1 and returns objective function value for that vector
            optimizer_name        (str) : optimizer name. For now, only qiskit optimizers are supported
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
        if contains_and_raised(problem_description, 'smooth_schedule'):
            self.obj_w = ObjectiveWrapperSmooth(obj, objective_parameters=objective_parameters, varform_description=varform_description, backend_description=backend_description, problem_description=problem_description, execute_parameters=execute_parameters)
        elif contains_and_raised(objective_parameters, 'use_mpo_energy'):
            from mpsbackend import ObjectiveWrapperMPOEnergy 
            self.obj_w = ObjectiveWrapperMPOEnergy(obj, objective_parameters=objective_parameters, varform_description=varform_description, backend_description=backend_description, problem_description=problem_description, execute_parameters=execute_parameters)
        else:
            self.obj_w = ObjectiveWrapper(obj, objective_parameters=objective_parameters, varform_description=varform_description, backend_description=backend_description, problem_description=problem_description, execute_parameters=execute_parameters)

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

    @abstractmethod
    def optimize(self):
        """Minimize the objective
        """

        return self.res


    def get_optimal_solution(self, shots=None, return_counts=False):
        """Returns the result of re-running QAOA with optimal angles found previously

        Args:
            shots (int) : number of shots to use
            return_counts (bool) : if raised, raw counts will be returned as well
        """
        final_execute_parameters = copy.deepcopy(self.execute_parameters)
        if self.backend_description['package'] == 'qiskit' and 'statevector' in self.backend_description['name']:
            sv = self.obj_w.var_form.run(self.res['opt_params'], backend_description=self.backend_description, execute_parameters=final_execute_parameters)
            sv_adj = get_adjusted_state(sv)
            counts = state_to_ampl_counts(sv_adj)
            assert(np.isclose(sum(np.abs(v)**2 for v in counts.values()), 1))
            objectives = [(self.obj(np.array([int(x) for x in k])), np.array([int(x) for x in k])) for k, v in counts.items() if (np.abs(v)**2) > 1e-5]
        else:
            if shots is not None:
                final_execute_parameters['shots'] = shots
            resstrs = self.obj_w.var_form.run(self.res['opt_params'], backend_description=self.backend_description, execute_parameters=final_execute_parameters)
            if return_counts:
                counts = Counter(''.join(str(i) for i in x) for x in resstrs)

            objectives = [(self.obj(x[::-1]), x) for x in resstrs]
        if return_counts:
            return min(objectives, key=itemgetter(0)), counts
        else:
            return min(objectives, key=itemgetter(0))
