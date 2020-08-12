""" ESCH (evolutionary algorithm). """

import logging
from qiskit.aqua.components.optimizers import Optimizer
from ._nloptimizer import minimize
from ._nloptimizer import check_pluggable_valid as check_nlopt_valid

logger = logging.getLogger(__name__)

try:
    import nlopt
except ImportError:
    logger.info('nlopt is not installed. Please install it if you want to use them.')


class BOBYQA(Optimizer):
    """BOBYQA.

    NLopt local optimizer, derivative-free
    https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/#bobyqa
    """

    CONFIGURATION = {
        'name': 'BOBYQA',
        'description': 'LN_BOBYQA Optimizer from nlopt',
        'input_schema': {
            '$schema': 'http://json-schema.org/draft-07/schema#',
            'id': 'bobyqa_schema',
            'type': 'object',
            'properties': {
                'max_evals': {
                    'type': 'integer',
                    'default': 1000
                },
                'ftol_rel': {
                    'type': 'number',
                    'default': 1e-4
                },
                'xtol_rel': {
                    'type': 'number',
                    'default': 1e-3
                }
            },
            'additionalProperties': False
        },
        'support_level': {
            'gradient': Optimizer.SupportLevel.ignored,
            'bounds': Optimizer.SupportLevel.supported,
            'initial_point': Optimizer.SupportLevel.required
        },
        'options': ['max_evals', 'ftol_rel', 'xtol_rel'],
        'optimizer': ['global']
    }

    def __init__(self, max_evals=1000, ftol_rel=1e-4, xtol_rel=1e-3):  # pylint: disable=unused-argument
        """
        Constructor.

        Args:
            max_evals (int): Maximum allowed number of function evaluations.
        """
        self.validate(locals())
        super().__init__()
        for k, v in locals().items():
            if k in self._configuration['options']:
                self._options[k] = v

    @staticmethod
    def check_pluggable_valid():
        check_nlopt_valid(BOBYQA.CONFIGURATION['name'])
        
    def get_support_level(self):
        return {
            'gradient': Optimizer.SupportLevel.ignored,
            'bounds': Optimizer.SupportLevel.ignored,
            'initial_point': Optimizer.SupportLevel.required
        }
        
    def optimize(self, num_vars, objective_function, gradient_function=None,
                 variable_bounds=None, initial_point=None):
        super().optimize(num_vars, objective_function,
                         gradient_function, variable_bounds, initial_point)

        return minimize(nlopt.LN_BOBYQA, objective_function, variable_bounds,
                        initial_point, **self._options)
