'''Setup scripts for Cortex.

'''
from .main import *
from . import manager


__version__ = '0.3a'
_manager = manager.get_manager()

methods = dict((method, getattr(_manager, method)) for method in dir(_manager)
 if callable(getattr(_manager, method))
 and not method.startswith('_'))

vars().update(**methods)