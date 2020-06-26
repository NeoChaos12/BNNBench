# Configure logging for module
import logging

logger = logging.getLogger(__name__)
sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
sh.setFormatter(logging.Formatter('[%(levelname)s] %(name)s at %(asctime)s --- %(message)s'))
logger.addHandler(sh)

from .base_model import BaseModel
from .bayesian_linear_regression import BayesianLinearRegression
from .mlp import MLP
from .dngo import DNGO
from .mcdropout import MCDropout
from .mcbatchnorm import MCBatchNorm
from .batchnorm import BatchNorm
from .deep_ensemble import DeepEnsemble

from pybnn.utils import AttrDict
model_types = AttrDict()
model_types.mlp = MLP
model_types.mcdropout = MCDropout
model_types.mcbatchnorm = MCBatchNorm
model_types.dngo = DNGO
model_types.ensemble = DeepEnsemble