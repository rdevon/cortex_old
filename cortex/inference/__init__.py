__all__ = ['air']

_classes = {}
from . import air
_modules = [air]
for module in _modules: _classes.update(**module._classes)