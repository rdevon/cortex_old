'''
Module for cortex namespaces.

.. note:: Credit to http://code.activestate.com/recipes/577887-a-simple-namespace-class/

'''

__all__ = ('Namespace',)

from collections import Mapping, Sequence


class Namespace(dict):
    '''A dict subclass that exposes its items as attributes.

    .. warning:: Namespace instances do not have direct access to the dict methods.

    '''

    def __init__(self, obj={}):
        dict.__init__(obj)

    def __dir__(self):
        return tuple(self)

    def __repr__(self):
        return '{0}({1})'.format(type(self).__name__, dict.__repr__(self))

    def __getattribute__(self, name):
        try:
            return self[name]
        except KeyError:
            msg = '`{0}` object has no attribute `{1}`'
            raise AttributeError(msg.format(type(self).__name__, name))

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]

    #------------------------ COPY CONSTRUCTORS

    @classmethod
    def from_object(cls, obj, names=None):
        if names is None:
            names = dir(obj)
        ns = {name:getattr(obj, name) for name in names}
        return cls(ns)

    @classmethod
    def from_mapping(cls, ns, names=None):
        if names:
            ns = {name:ns[name] for name in names}
        return cls(ns)

    @classmethod
    def from_sequence(cls, seq, names=None):
        if names:
            seq = {name:val for name, val in seq if name in names}
        return cls(seq)

    #------------------------ STATIC METHODS

    @staticmethod
    def hasattr(ns, name):
        try:
            object.__getattribute__(ns, name)
        except AttributeError:
            return False
        return True

    @staticmethod
    def getattr(ns, name):
        return object.__getattribute__(ns, name)

    @staticmethod
    def setattr(ns, name, value):
        return object.__setattr__(ns, name, value)

    @staticmethod
    def delattr(ns, name):
        return object.__delattr__(ns, name)
