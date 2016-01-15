'''
Module for Theano probabilistic distributions.
'''

import theano
from theano import tensor as T

from utils.tools import (
    e,
    floatX,
    pi,
    _slice
)


# BERNOULLI --------------------------------------------------------------------

def _binomial(trng, p, size=None):
    if size is None:
        size = p.shape
    return trng.binomial(p=p, size=size, n=1, dtype=p.dtype)

def _centered_binomial(trng, p, size=None):
    if size is None:
        size = p.shape
    return 2 * trng.binomial(p=0.5*(p+1), size=size, n=1, dtype=p.dtype) - 1.

def _cross_entropy(x, p, axis=None, scale=1.0):
    p = T.clip(p, 1e-7, 1.0 - 1e-7)
    energy = T.nnet.binary_crossentropy(p, x)
    if axis is None:
        axis = energy.ndim - 1
    energy = energy.sum(axis=axis)
    return (scale * energy).astype('float32')

def _binary_entropy(p, axis=None):
    p_c = T.clip(p, 1e-7, 1.0 - 1e-7)
    entropy = T.nnet.binary_crossentropy(p_c, p)
    if axis is None:
        axis = entropy.ndim - 1
    entropy = entropy.sum(axis=axis)
    return entropy

# SOFTMAX ----------------------------------------------------------------------

def _softmax(x, axis=None):
    if axis is None:
        axis = x.ndim - 1
    e_x = T.exp(x - x.max(axis=axis, keepdims=True))
    out = e_x / e_x.sum(axis=axis, keepdims=True)
    return out

def _sample_softmax(trng, p, size=None):
    if size is None:
        size = p.shape
    return trng.multinomial(pvals=p, size=size).astype('float32')

def _categorical_cross_entropy(x, p, axis=None, scale=1.0):
    p = T.clip(p, 1e-7, 1.0 - 1e-7)
    #energy = T.nnet.categorical_crossentropy(p, x)
    energy = T.nnet.binary_crossentropy(p, x)
    if axis is None:
        axis = x.ndim - 1
    energy = energy.sum(axis=axis)
    return (scale * energy / p.shape[p.ndim-1]).astype('float32')

def _categorical_entropy(p, axis=None):
    p_c = T.clip(p, 1e-7, 1.0 - 1e-7)
    entropy = T.nnet.categorical_crossentropy(p_c, p)
    return entropy

# GAUSSIAN ---------------------------------------------------------------------

def _normal(trng, p, size=None):
    dim = p.shape[p.ndim-1] // 2
    mu = _slice(p, 0, dim)
    log_sigma = _slice(p, 1, dim)

    if size is None:
        size = mu.shape
    return trng.normal(avg=mu, std=T.exp(log_sigma), size=size)

def _normal_prob(p):
    dim = p.shape[p.ndim-1] // 2
    mu = _slice(p, 0, dim)
    return mu

def _neg_normal_log_prob(x, p, axis=None, scale=1.0):
    dim = p.shape[p.ndim-1] // 2
    mu = _slice(p, 0, dim)
    log_sigma = _slice(p, 1, dim)
    energy = 0.5 * (
        (x - mu)**2 / (T.exp(2 * log_sigma)) + 2 * log_sigma + T.log(2 * pi))

    if axis is None:
        axis = energy.ndim - 1
    energy = energy.sum(axis=axis)
    return (scale * energy).astype('float32')

def _normal_entropy(p, axis=None):
    dim = p.shape[p.ndim-1] // 2
    log_sigma = _slice(p, 1, dim)

    entropy = 0.5 * T.log(2 * pi * e) + log_sigma
    if axis is None:
        axis = entropy.ndim - 1
    entropy = entropy.sum(axis=axis)
    return entropy