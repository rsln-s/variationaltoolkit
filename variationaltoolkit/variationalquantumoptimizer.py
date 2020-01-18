import numpy as np
import copy
from operator import itemgetter
import variationaltoolkit.optimizers as vt_optimizers
import qiskit.aqua.components.optimizers as qiskit_optimizers
from .objectivewrapper import ObjectiveWrapper

class VariationalQuantumOptimizer:
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
            problem_description (dict)  : See varform.py 
            backend_description (dict)  : See varform.py
            execute_parameters (dict)   : See objectivewrapper.py
            objective_parameters (dict) : See objectivewrapper.py 
        """
        self.obj = obj
        self.obj_w = ObjectiveWrapper(obj, objective_parameters=objective_parameters, varform_description=varform_description, backend_description=backend_description, problem_description=problem_description, execute_parameters=execute_parameters)

        self.optimizer_name = optimizer_name
        self.optimizer_parameters = optimizer_parameters
        # Check local variationaltoolkit optimizers first
        if hasattr(vt_optimizers, self.optimizer_name):
            optimizer_namespace = vt_optimizers
        elif hasattr(qiskit_optimizers, self.optimizer_name):
            # fallback on qiskit
            optimizer_namespace = qiskit_optimizers
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        self.optimizer = getattr(optimizer_namespace, self.optimizer_name)(**self.optimizer_parameters)

        if variable_bounds is not None:
            self.variable_bounds = variable_bounds
        else:
            self.variable_bounds = [(None, None)] * self.obj_w.var_form.num_parameters 
        if initial_point is not None:
            self.initial_point=initial_point
        else:
            lb = [(l if l is not None else -2 * np.pi) for (l, u) in self.variable_bounds]
            ub = [(u if u is not None else 2 * np.pi) for (l, u) in self.variable_bounds]
            self.initial_point = np.random.uniform(lb,ub, self.obj_w.var_form.num_parameters)
        self.varform_description = varform_description
        self.problem_description = problem_description
        self.execute_parameters = execute_parameters
        self.objective_parameters = objective_parameters
        self.backend_description = backend_description
        self.res = {}

    def optimize(self):
        """Minimize the objective
        """
        opt_params, opt_val, num_optimizer_evals = self.optimizer.optimize(self.obj_w.var_form.num_parameters, 
                                                                      self.obj_w.get_obj(), 
                                                                      variable_bounds = self.variable_bounds,
                                                                      initial_point = self.initial_point)
        self.res['num_optimizer_evals'] = num_optimizer_evals
        self.res['min_val'] = opt_val
        self.res['opt_params'] = opt_params

        return self.res

    def get_optimal_solution(self, shots=None):
        """
        TODO: should support running separately on device
        Returns minimal(!!) energy string
        """
        final_execute_parameters = copy.deepcopy(self.execute_parameters)
        if shots is not None:
            final_execute_parameters['shots'] = shots
        resstrs = self.obj_w.var_form.run(self.res['opt_params'], backend_description=self.backend_description, execute_parameters=final_execute_parameters)

        objectives = [(self.obj(x), x) for x in resstrs]
        return min(objectives, key=itemgetter(0))
