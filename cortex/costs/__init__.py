__all__ = ['basic', 'decay', 'stats']

from .basic import *

_costs = {}
_stats = {}

from . import basic, decay, stats
modules = [basic, decay, stats]
for module in modules:
    try:
        _costs.update(**module._costs)
    except AttributeError:
        pass
    try:
        _stats.update(**module._stats)
    except AttributeError:
        pass