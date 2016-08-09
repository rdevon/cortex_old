__all__ = ['mlp', 'distribution_mlp', 'cnn2d']

from .mlp import *
from . import mlp, distribution_mlp, cnn2d


_classes = {}
_modules = [mlp, distribution_mlp, cnn2d]
for module in _modules:
    _classes.update(**module._classes)