__all__ = ['dataset', 'basic', 'neuroimaging']

from dataset import *

_classes = {}
from . import dataset, basic, neuroimaging
_modules = [dataset, basic, neuroimaging]
for module in _modules: _classes.update(**module._classes)