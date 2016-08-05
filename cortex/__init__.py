'''Setup scripts for Cortex.

'''
import logging
import readline, glob
from os import path
import urllib2

from .main import *
from . import manager
from .utils.tools import get_paths, _p
from .utils.extra import (
    complete_path, query_yes_no, write_default_theanorc, write_path_conf)
from .utils import logger as cortex_logger


cortex_logger.set_stream_logger(1)

__version__ = '0.3a'
logger = logging.getLogger(__name__)
_manager = manager.get_manager()

methods = dict((method, getattr(_manager, method)) for method in dir(_manager)
 if callable(getattr(_manager, method))
 and not method.startswith('_'))

vars().update(**methods)

