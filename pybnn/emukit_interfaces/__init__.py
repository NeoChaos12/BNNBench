import logging

_log = logging.getLogger(__name__)
# _log.setLevel(logging.WARNING)

from pybnn.emukit_interfaces.hpobench import Benchmarks, HPOBenchObjective
from pybnn.emukit_interfaces.models import SciPyLikeModelWrapper, PyBNNModel
from pybnn.emukit_interfaces.synthetic_objectives import branin, hartmann3_2, borehole_6