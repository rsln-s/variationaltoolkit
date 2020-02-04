import numpy as np
import logging
from .varform import VarForm
from .utils import validate_objective, contains_and_raised, state_to_ampl_counts, obj_from_statevector

class ObjectiveWrapper:
    """Objective Function Wrapper
    
    Wraps variational form object and an objective into something that can be passed to an optimizer.
    Remembers all the previous values.
    """

    def __init__(self, obj, objective_parameters=None, varform_description=None, backend_description=None, problem_description=None, execute_parameters=None):
        """Constuctor.
        
        Args:
            obj (function) : takes a list of 0,1 and returns objective function value for that vector
            varform_description (dict) : See varform.py
            problem_description (dict) : See varform.py 
            backend_description (dict) : See varform.py
            execute_parameters (dict)  : Parameters passed to execute function (e.g. {'shots': 8000})
            objective_parameters (dict)  : Parameters for objective function. 
                                           Accepted fields:
                                           'save_vals' (bool) -- save values of the objective function 
                                           Note: statistic on the value of objective function (e.g. mean) is saved automatically
                                           'save_resstrs' (bool) -- save all raw resstrs
        """
        validate_objective(obj, varform_description['num_qubits'])

        self.obj = obj
        self.varform_description = varform_description
        self.problem_description = problem_description
        self.execute_parameters = execute_parameters
        self.objective_parameters = objective_parameters
        self.backend_description = backend_description
        if 'package' in self.varform_description and self.varform_description['package'] == 'mpsbackend':
            import mpsbackend.variational_forms as mps_variational_forms
            varform_parameters = {k : v for k,v in varform_description.items() if k != 'name' and k != 'package'}
            self.var_form = getattr(mps_variational_forms, varform_description['name'])(**varform_parameters)
        else:
            self.var_form = VarForm(varform_description=varform_description, problem_description=problem_description)
        self.vals_statistic = []
        self.vals = []
        self.points = []
        self.resstrs = []


    def get_obj(self):
        """Returns objective function
        """
        def f(theta):
            self.points.append(theta)
            resstrs = self.var_form.run(theta, backend_description=self.backend_description, execute_parameters=self.execute_parameters)
            if self.backend_description['package'] == 'qiskit' and 'statevector' in self.backend_description['name']:
                objective_value = obj_from_statevector(resstrs, self.obj)
            else:
                if contains_and_raised(self.objective_parameters, 'save_resstrs'):
                    self.resstrs.append(resstrs)

                vals = [self.obj(x) for x in resstrs] 
                if contains_and_raised(self.objective_parameters, 'save_vals'):
                    self.vals.append(vals)

                # TODO: should allow for different statistics (e.g. CVAR)
                objective_value = np.mean(vals) 
            logging.info(f"called at step {len(self.vals_statistic)}, objective: {objective_value} at point {theta}")
            self.vals_statistic.append(objective_value)

            return objective_value

        return f
