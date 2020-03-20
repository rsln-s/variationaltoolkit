import logging
from .varform import VarForm
from .variationalquantumoptimizersequential import VariationalQuantumOptimizerSequential
from .objectivewrapper import ObjectiveWrapper

try:
    from .variationalquantumoptimizeraposmm import VariationalQuantumOptimizerAPOSMM
except ImportError as e:
    logging.warning(f"Failed loading VariationalQuantumOptimizerAPOSMM, ignoring the following error: '{e}'")
    

