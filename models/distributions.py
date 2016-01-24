'''
Module for Theano probabilistic distributions.
'''

from collections import OrderedDict
import numpy as np
import theano
from theano import tensor as T

from models.layers import Layer
from utils.tools import (
    e,
    floatX,
    init_rngs,
    init_weights,
    pi,
    _slice
)


class Distribution(Layer):
    def __init__(self, dim, name='distribution', **kwargs):
        self.dim = dim

        kwargs = init_weights(self, **kwargs)
        kwargs = init_rngs(self, **kwargs)

        super(Distribution, self).__init__(name=name)

    def set_params(self):
        raise NotImplementedError()

    def get_prob(self):
        raise NotImplementedError()

    def sample(self, size=None):
        p = self.get_prob()
        return self.f_sample(self.trng, p, size=size)

    def neg_log_prob(self, x, axis=None, scale=None):
        p = self.get_prob()
        return self.f_neg_log_prob(x, p, axis=None, scale=None)

    def entropy(self, axis=None):
        p = self.get_prob()
        return self.f_entropy(p, axis=None)


class Bernoulli(Distribution):
    def __init__(self, dim, name='bernoulli', **kwargs):
        self.f_sample = _binomial
        self.f_neg_log_prob = _cross_entropy
        self.f_entropy = _binary_entropy
        super(Bernoulli, self).__init__(dim, name=name, **kwargs)

    def set_params(self):
        z = np.zeros((self.dim,)).astype(floatX)
        self.params = OrderedDict(z=z)

    def get_params(self):
        return [T.nnet.sigmoid(self.z)]

    def get_prob(self):
        return T.nnet.sigmoid(self.z)


class Gaussian(Distribution):
    def __init__(self, dim, name='gaussian', **kwargs):
        self.f_sample = _normal
        self.f_neg_log_prob = _neg_normal_log_prob
        self.f_entropy = _normal_entropy
        super(Gaussian, self).__init__(dim, name=name, **kwargs)

    def set_params(self):
        mu = np.zeros((self.dim_h,)).astype(floatX)
        log_sigma = np.zeros((self.dim_h,)).astype(floatX)

        self.params = OrderedDict(
            mu=mu, log_sigma=log_sigma)

    def get_params(self):
        return [self.mu, self.log_sigma]

    def get_prob(self, ):
        return concatenate([self.mu, self.log_sigma])


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