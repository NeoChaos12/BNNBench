import logging

logger = logging.getLogger(__name__)
sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
sh.setFormatter(logging.Formatter('[%(levelname)s] %(name)s at %(asctime)s --- %(message)s'))
logger.addHandler(sh)

loggers = {"global": logger}

import pybnn.utils as utils
import pybnn.models as models

loggers["model"] = models.logger
loggers["util"] = utils.logger
#TODO: Remove this log message after testing
logger.info("Package PyBNN initialized.")