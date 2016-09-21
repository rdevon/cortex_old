'''Setup scripts for Cortex.

'''
try:
    import gym
except ImportError:
    pass
import logging
import warnings

from .main import *
from . import manager


__version__ = '0.3a'
_manager = manager.get_manager()

#assert False, logging.Logger.manager.loggerDict['gym'].keys()
logging.Logger.manager.loggerDict['gym'].setLevel(logging.ERROR)
warnings.filterwarnings('ignore', module='nipy')
warnings.filterwarnings('ignore', module='gym')
warnings.filterwarnings('ignore',
                        'This call to matplotlib.use() has no effect.*',
                        UserWarning)

methods = dict((method, getattr(_manager, method)) for method in dir(_manager)
 if callable(getattr(_manager, method))
 and not method.startswith('_'))

vars().update(**methods)