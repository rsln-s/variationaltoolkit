import numpy as np
import logging
import copy
from operator import itemgetter
import variationaltoolkit.optimizers as vt_optimizers
import qiskit.aqua.components.optimizers as qiskit_optimizers
from collections import Counter
from .objectivewrapper import ObjectiveWrapper
from .variationalquantumoptimizer import VariationalQuantumOptimizer
from .objectivewrappersmooth import ObjectiveWrapperSmooth
from .utils import state_to_ampl_counts, check_cost_operator, get_adjusted_state, contains_and_raised

logger = logging.getLogger(__name__)

class VariationalQuantumOptimizerSequential(VariationalQuantumOptimizer):
    def __init__(self, obj, optimizer_name, **kwargs):
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
        super().__init__(obj, optimizer_name, **kwargs)

        self.optimizer_name = optimizer_name
        self.optimizer_parameters = kwargs['optimizer_parameters']
        # Check local variationaltoolkit optimizers first
        if hasattr(vt_optimizers, self.optimizer_name):
            optimizer_namespace = vt_optimizers
        elif hasattr(qiskit_optimizers, self.optimizer_name):
            # fallback on qiskit
            optimizer_namespace = qiskit_optimizers
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")
        self.optimizer = getattr(optimizer_namespace, self.optimizer_name)(**self.optimizer_parameters)


    def optimize(self):
        """Minimize the objective
        """
        opt_params, opt_val, num_optimizer_evals = self.optimizer.optimize(self.obj_w.num_parameters, 
                                                                      self.obj_w.get_obj(), 
                                                                      variable_bounds = self.variable_bounds,
                                                                      initial_point = self.initial_point)
        self.res['num_optimizer_evals'] = num_optimizer_evals
        self.res['min_val'] = opt_val
        self.res['opt_params'] = opt_params

        return self.res

