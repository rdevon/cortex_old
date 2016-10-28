__all__ = ['air']

_classes = {}
from . import air, gdir
_modules = [air, gdir]
for module in _modules: _classes.update(**module._classes)