import numpy as np
import logging
from .objectivewrapper import ObjectiveWrapper
from .utils import contains_and_raised, state_to_ampl_counts, obj_from_statevector

class ObjectiveWrapperSmooth(ObjectiveWrapper):
    """A class to wrap smooth QAOA schedules

    Only supports QAOA
    """

    def __init__(self, *args, **kwargs):
        if kwargs['varform_description']['name'] != 'QAOA':
            raise ValueError(f"Smooth schedules are only support for QAOA varform, received {varform_description['name']}")
        kwargs['varform_description'].pop('smooth_schedule') # kinda hacky
        super().__init__(*args, **kwargs)
        self.num_parameters = 4 
        self.p = self.varform_description['p']

    def get_obj(self):
        """Returns objective function

        The objective function takes  the ends  of beta and gamma and extrapolates them linearly
        """
        def f(theta):
            assert(len(theta) == 4)
            beta_0 = theta[0]
            beta_1 = theta[1]
            gamma_0 = theta[2]
            gamma_1 = theta[3]
            theta = np.hstack([np.linspace(beta_0, beta_1, self.p), np.linspace(gamma_0, gamma_1, self.p)])
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
