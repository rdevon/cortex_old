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

try:
    logging.Logger.manager.loggerDict['gym'].setLevel(logging.ERROR)
    warnings.filterwarnings('ignore', module='gym')
except:
    pass
    
warnings.filterwarnings('ignore', module='nipy')
warnings.filterwarnings('ignore',
                        'This call to matplotlib.use() has no effect.*',
                        UserWarning)
