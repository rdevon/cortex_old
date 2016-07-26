'''Module for Multinomial distributions.

'''

from collections import OrderedDict
import numpy as np
from theano import tensor as T

from . import Distribution
from ...utils import floatX


def _softmax(x):
    axis = x.ndim - 1
    e_x = T.exp(x - x.max(axis=axis, keepdims=True))
    out = e_x / e_x.sum(axis=axis, keepdims=True)
    return out

def _sample_multinomial(trng, p, size=None):
    if size is None:
        size = p.shape
    return trng.multinomial(pvals=p, size=size).astype(floatX)

def _categorical_cross_entropy(x, p, sum_probs=True):
    p = T.clip(p, _clip, 1.0 - _clip)
    energy = T.nnet.binary_crossentropy(p, x)
    if sum_probs:
        return energy.sum(axis=x.ndim-1)
    else:
        return energy

def _categorical_entropy(p):
    p_c = T.clip(p, _clip, 1.0 - _clip)
    entropy = T.nnet.categorical_crossentropy(p_c, p)
    return entropy


class Multinomial(Distribution):
    '''Multinomial distribuion.

    '''

    def __init__(self, dim, name='multinomial', **kwargs):
        self.f_sample = _sample_multinomial
        self.f_neg_log_prob = _categorical_cross_entropy
        self.f_entropy = _categorical_entropy
        super(Multinomial, self).__init__(dim, name=name, **kwargs)

    def _act(self, X):
        return _softmax(X)

    def init_params(self):
        z = np.zeros((self.dim,)).astype(floatX)
        self.params = OrderedDict(z=z)


_classes = {'multinomial': Multinomial}