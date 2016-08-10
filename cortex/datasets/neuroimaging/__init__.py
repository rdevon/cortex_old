__all__ = ['ni_dataset']

from ni_dataset import *

_classes = {}
from . import mri, fmri
_modules = [mri, fmri]
for module in _modules: _classes.update(**module._classes)