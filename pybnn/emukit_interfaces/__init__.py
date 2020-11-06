import logging

_log = logging.getLogger(__name__)
# _log.setLevel(logging.WARNING)

from pybnn.emukit_interfaces.hpolib_benchmarks import Benchmarks, HPOlibBenchmarkObjective
from pybnn.emukit_interfaces.models import SciPyLikeModelWrapper, PyBNNModel