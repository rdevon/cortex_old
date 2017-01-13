'''Module for Multinomial distributions.

'''

from collections import OrderedDict
import numpy as np
from theano import tensor as T

from . import Distribution, _clip
from ...utils import floatX, intX, scan


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

    def quantile(self, E, P):
        assert E.ndim == (P.ndim - 1)
        if P.ndim == 2:
            shape = None
        elif P.ndim == 3:
            shape = P.shape
            P = P.reshape((P.shape[0] * P.shape[1], P.shape[2]))
            E = E.reshape((E.shape[0] * E.shape[1],))
        elif P.ndim == 4:
            shape = P.shape
            P = P.reshape((P.shape[0] * P.shape[1] * P.shape[2], P.shape[3]))
            E = E.reshape((E.shape[0] * E.shape[1] * E.shape[2],))
        else:
            raise NotImplementedError()
        
        P_e = T.tile(P[:, :, None], P.shape[1])
        
        def step(A):
            return T.triu(A)
        
        tria, _ = scan(step, [P_e], [None], [], P_e.shape[0])
        sums = tria.sum(axis=1)
        
        # This may not make sense, but we add the upper trangular rows to get
        # each action's cumulative, then subtract epsilon. We want the
        # lowest non-negative number.
        diff = sums - E[:, None]
        diff = T.switch(T.le(diff, 0.), diff.max() + 1., diff).astype(floatX)
        S = diff.argmin(axis=1).astype(intX)
        S = T.extra_ops.to_one_hot(S, P.shape[-1])
        #S = self.trng.multinomial(pvals=P).astype(floatX)
        if shape is not None: S = S.reshape(shape)
        return S
    
    def generate_random_variables(self, shape, P=None):
        return self.random_variables(shape)

    def random_variables(self, size):
        return self.trng.uniform(size, dtype=floatX)

_classes = {'multinomial': Multinomial}