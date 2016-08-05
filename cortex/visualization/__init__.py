from ..utils.tools import _p

__all__ = ['classifier']

_ops = {}
from . import basic, classifier
modules = [basic, classifier]
for module in modules:
    ops = module._ops
    name = '.'.join(module.__name__.split('.')[1:])
    if name == 'visualization.basic':
        name  = 'visualization'
    ops = dict((_p(name, k), v) for k, v in ops.iteritems())
    _ops.update(ops)