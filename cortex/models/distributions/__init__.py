__all__ = ['distribution', 'binomial', 'gaussian', 'laplace', 'multinomial',
           'logistic']

from distribution import *

_clip = 1e-7

_classes = {}
from . import binomial, gaussian, laplace, logistic, multinomial
_modules =[binomial, gaussian, laplace, logistic, multinomial]
for module in _modules: _classes.update(**module._classes)
_conditionals = {}

keys = _classes.keys()
for k in keys:
    v = _classes[k]
    C = make_conditional(v)
    _conditionals[v.__name__] = C
    _classes['conditional_' + k] = C