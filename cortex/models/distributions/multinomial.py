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
        Pdim = P.ndim
        if Pdim == 2:
            shape = None
            if P.ndim == (E.ndim + 1):
                fshape = P.shape
            else:
                fshape = (E.shape[0], P.shape[0], P.shape[1])
        elif Pdim == 3:
            shape = P.shape
            if P.ndim == (E.ndim + 1):
                fshape = shape
            else:
                fshape = (E.shape[0], P.shape[0], P.shape[1], P.shape[2])
            P = P.reshape((P.shape[0] * P.shape[1], P.shape[2]))
        elif Pdim == 4:
            shape = P.shape
            if P.ndim == (E.ndim + 1):
                fshape = shape
            else:
                fshape = (E.shape[0], P.shape[0], P.shape[1], P.shape[2], P.shape[3])
            P = P.reshape((P.shape[0] * P.shape[1] * P.shape[2], P.shape[3]))
        else:
            raise NotImplementedError()
    
        def step(i, P):
            A = T.zeros((P.shape[-1],))
            A = T.set_subtensor(A[:i+1], 1.).astype(floatX)
            return (P * A[None, :]).sum(-1)
        
        sums, _ = scan(step, [T.arange(P.shape[-1])], [None], [P], P.shape[-1])
        sums = sums.T
        
        # This may not make sense, but we add the upper trangular rows to get
        # each action's cumulative, then subtract epsilon. We want the
        # lowest non-negative number.
        
        if shape is not None:
            sums = sums.reshape(shape)
        
        if sums.ndim == E.ndim:
            sums = T.shape_padleft(sums)
        E = T.shape_padright(E)
            
        diff = sums - E
        diff = diff.flatten()
        diff = diff.reshape((diff.shape[0] // P.shape[-1], P.shape[-1]))
        diff = T.switch(T.le(diff, 0.), diff.max() + 1., diff).astype(floatX)
        S = diff.argmin(axis=1).astype(intX)
        S = T.extra_ops.to_one_hot(S, P.shape[-1])
        S = S.reshape(fshape)
        return S
    
    def generate_random_variables(self, shape, P=None):
        shape = tuple(shape)
        if P is None:
            pass
        elif P.ndim == 1:
            shape = shape
        elif P.ndim == 2:
            shape = shape + (P.shape[0],)
        elif P.ndim == 3:
            shape = shape + (P.shape[0], P.shape[1], )
        elif P.ndim == 4:
            shape = shape + (P.shape[0], P.shape[1], P.shape[2])
        else:
            raise ValueError(P.ndim)
        return self.random_variables(shape)

    def random_variables(self, size):
        S = self.trng.uniform(size, dtype=floatX)
        return S

_classes = {'multinomial': Multinomial}