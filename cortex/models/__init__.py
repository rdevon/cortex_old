'''
Base Layer class
'''

from collections import OrderedDict
import copy
import theano

from cortex.utils.tools import (
    warn_kwargs,
    _p
)


class Layer(object):
    def __init__(self, name='', excludes=[], learn=True, **kwargs):
        self.name = name
        self.params = None
        self.excludes = excludes
        self.learn = learn
        self.set_params()
        self.n_params = len(self.params)
        warn_kwargs(kwargs)

    def copy(self):
        return copy.deepcopy(self)

    def set_params(self):
        raise NotImplementedError()

    def set_tparams(self):
        if self.params is None:
            raise ValueError('Params not set yet')
        tparams = OrderedDict()
        for kk, pp in self.params.iteritems():
            tp = theano.shared(self.params[kk], name=kk)
            tparams[_p(self.name, kk)] = tp
            self.__dict__[kk] = tp

        return OrderedDict((k, v) for k, v in tparams.iteritems() if k not in [_p(self.name, e) for e in self.excludes])

    def get_excludes(self):
        if self.learn:
            return [_p(self.name, e) for e in self.excludes]
        else:
            return [_p(self.name, k) for k in self.params.keys()]

    def __call__(self, state_below):
        raise NotImplementedError()