import logging

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.WARNING)

from pybnn.emukit_interfaces.hpolib_benchmarks import Benchmarks, HPOlibBenchmarkObjective
from pybnn.emukit_interfaces.models import SciPyLikeModelWrapper, PyBNNModel