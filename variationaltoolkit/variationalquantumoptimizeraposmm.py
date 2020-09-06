import numpy as np
import logging
import copy
from operator import itemgetter
import variationaltoolkit.optimizers as vt_optimizers
import qiskit.aqua.components.optimizers as qiskit_optimizers

from mpi4py import MPI  # for libE communicator
import os  # for adding to path
from libensemble.libE import libE
from libensemble.gen_funcs.persistent_aposmm import aposmm as gen_f
from libensemble.alloc_funcs.persistent_aposmm_alloc import persistent_aposmm_alloc as alloc_f
from libensemble.tools import parse_args, add_unique_random_streams

from .objectivewrapper import ObjectiveWrapper
from .variationalquantumoptimizer import VariationalQuantumOptimizer
from .objectivewrappersmooth import ObjectiveWrapperSmooth
from .utils import state_to_ampl_counts, check_cost_operator, get_adjusted_state, contains_and_raised

logger = logging.getLogger(__name__)

nworkers, is_master, libE_specs, _ = parse_args()
libE_specs['save_H_and_persis_on_abort'] = False
libE_specs['disable_log_files'] = True

class VariationalQuantumOptimizerAPOSMM(VariationalQuantumOptimizer):
    def __init__(self, obj, optimizer_name, gen_specs_user={}, **kwargs):
        """Constuctor.
        
        Args:
            obj (function) : takes a list of 0,1 and returns objective function value for that vector
            optimizer_name        (str) : optimizer name. For now, only qiskit optimizers are supported
            gen_specs_user       (dict) : passed directly to libE via gen_specs['user']
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
        if self.optimizer_name in ['scipy_Nelder-Mead', 'scipy_COBYLA']:
            if 'options' in self.optimizer_parameters:
                # assume scipy kwargs
                gen_specs_user['scipy_kwargs'] = self.optimizer_parameters
                self.exit_criteria = {'sim_max': self.optimizer_parameters['options']['maxiter']}
            else:
                # use default
                if 'maxiter' in self.optimizer_parameters:
                    _maxiter = self.optimizer_parameters['maxiter']
                else:
                    _maxiter = 100 
                    logger.warning(f"Ignoring scipy parameters -- incorrect format received: {self.optimizer_parameters}")
                gen_specs_user['scipy_kwargs'] = {'tol': 1e-10, 'options': {'disp':False, 'maxiter': _maxiter}}
                self.exit_criteria = {'sim_max': _maxiter}
        else:
            # assume nlopt
            if len(gen_specs_user) == 0:
                raise ValueError(f"When using nlopt, gen_user_spec must specify tolerances")
            # Tell libEnsemble when to stop
            self.exit_criteria = {'sim_max': self.optimizer_parameters['maxiter']}
        self.gen_specs_user = gen_specs_user


    def optimize(self):
        """Minimize the objective
        """
        obj_val = self.obj_w.get_obj()
        num_parameters = self.obj_w.num_parameters
        lb = np.array([(l if l is not None else -2 * np.pi) for (l, u) in self.variable_bounds])
        ub = np.array([(u if u is not None else 2 * np.pi) for (l, u) in self.variable_bounds])

        def sim_func(H, gen_info, sim_specs, libE_info):
            del libE_info  # Ignored parameter

            batch = len(H['x'])
            O = np.zeros(batch, dtype=sim_specs['out'])

            for i, x in enumerate(H['x']):
                O['f'][i] = obj_val(x)

            return O, gen_info

        script_name = os.path.splitext(os.path.basename(__file__))[0]
        #State the objective function, its arguments, output, and necessary parameters (and their sizes)
        sim_specs = {
            'sim_f':
                sim_func,  # This is the function whose output is being minimized
            'in': ['x'],  # These keys will be given to the above function
            'out': [
                ('f',
                 float),  # This is the output from the function being minimized
            ],
        }
        gen_out = [
            ('x', float, num_parameters),
            ('x_on_cube', float, num_parameters),
            ('sim_id', int),
            ('local_pt', bool),
            ('local_min', bool),
  ]

        np.random.seed(0)
        # State the generating function, its arguments, output, and necessary parameters.
        # This is the default dictionary, we update it below
        gen_specs = {
            'gen_f': gen_f,
            'in': ['x', 'f', 'local_pt', 'sim_id', 'returned', 'x_on_cube', 'local_min'],
            #'mu':0.1,   # limit on dist_to_bound: everything closer to bound than mu is thrown out
            'out': gen_out,
            'user':{
                'lb': lb,
                'ub': ub,
                'initial_sample_size': nworkers-1,  # num points sampled before starting opt runs, one per worker
                'localopt_method': self.optimizer_name,
                'sample_points': np.atleast_2d(copy.deepcopy(self.initial_point)),
                'run_max_eval':100,
                'num_pts_first_pass': nworkers-1,
                'max_active_runs': 2,
                'periodic': self.obj_w.is_periodic,
            }
        }
        gen_specs['user'].update(self.gen_specs_user)

        persis_info = add_unique_random_streams({}, nworkers + 1)

        alloc_specs = {'alloc_f': alloc_f, 'out': [('given_back', bool)], 'user': {}}

        H, persis_info, flag = libE(
            sim_specs,
            gen_specs,
            self.exit_criteria,
            persis_info=persis_info,
            libE_specs=libE_specs,
            alloc_specs=alloc_specs)

        if MPI.COMM_WORLD.Get_rank() == 0:
            self.res['num_optimizer_evals'] = len(H['f'])
            min_index = np.argmin(H['f'])
            self.res['min_val'] = H['f'][min_index]
            self.res['opt_params'] = H['x'][min_index]
            self.res['H'] = H
            self.res['persis_info'] = persis_info
            return self.res
