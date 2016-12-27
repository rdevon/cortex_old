__all__ = ['dataset', 'basic', 'neuroimaging']

from dataset import *

_classes = {}
from . import dataset, basic
_modules = [dataset, basic]
for module in _modules: _classes.update(**module._classes)