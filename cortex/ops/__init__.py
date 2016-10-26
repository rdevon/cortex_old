from ..utils.tools import _p

__all__ = ['noise']

_ops = {}
from . import noise, op
modules = [noise, op]
for module in modules:
    ops = module._ops
    name = '.'.join(module.__name__.split('.')[2:])
    ops = dict((_p(name, k), v) for k, v in ops.iteritems())
    _ops.update(ops)
