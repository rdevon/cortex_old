'''Namespace for objects in cortex.

Used for datasets and cells.

'''

from . import Namespace
from ..base import Base


class ObjectNamespace(Namespace):
    '''Namespace for datasets in cortex.
    
    '''
    
    def __setattr__(self, name, value): 
        
        if not isinstance(value, Base):
            raise ValueError('Only cortex objects allowed in ObjectNamespace')
        self[name] = value