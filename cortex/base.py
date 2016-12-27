'''Base cortex class.

'''

import manager
from .utils.logger import get_class_logger


class Base(object):
    def __init__(self, name=None):
        if not hasattr(self, 'logger'): self.logger = get_class_logger(self)
        if name is None: raise ValueError()
        self.name = name
        self.manager = manager
        
        try:
            self.manager.objects[self.name]
            raise ValueError('Object with name `{}` already exists. Use the '
                             '`name` argument to assign a unique id'
                             ''.format(self.name))
        except KeyError:
            pass
        self.manager.objects[self.name] = self
        