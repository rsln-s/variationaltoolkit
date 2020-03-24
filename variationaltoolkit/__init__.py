import logging
from .varform import VarForm
from .variationalquantumoptimizersequential import VariationalQuantumOptimizerSequential
from .objectivewrapper import ObjectiveWrapper

root_logger = logging.getLogger('variationaltoolkit')
root_logger.setLevel(logging.WARN)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s')
ch.setFormatter(formatter)
root_logger.addHandler(ch)

logger = logging.getLogger(__name__)

try:
    from .variationalquantumoptimizeraposmm import VariationalQuantumOptimizerAPOSMM
except ImportError as e:
    logger.warn(f"Failed loading VariationalQuantumOptimizerAPOSMM, ignoring the following error: '{e}'")
    

