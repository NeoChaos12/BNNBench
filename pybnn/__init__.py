import logging

logger = logging.getLogger(__name__)
sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
sh.setFormatter(logging.Formatter('[%(levelname)s] %(name)s at %(asctime)s --- %(message)s'))
logger.addHandler(sh)


import pybnn.utils as utils
import pybnn.models as models

logger.info("Package PyBNN initialized.")