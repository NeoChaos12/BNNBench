import logging

logger = logging.getLogger(__name__)
sh = logging.StreamHandler()
sh.setLevel(logging.DEBUG)
sh.setFormatter(logging.Formatter('[%(levelname)s] %(name)s at %(asctime)s --- %(message)s'))
logger.addHandler(sh)

from .attrDict import AttrDict
from .universal_utils import dict_fetch
from .data_utils import dataloader_args