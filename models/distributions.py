'''
Module for Theano probabilistic distributions.
'''

from collections import OrderedDict
import numpy as np
import theano
from theano import tensor as T

from models.layers import Layer
from utils.tools import (
    concatenate,
    e,
    floatX,
    init_rngs,
    init_weights,
    pi,
    _slice
)


_clip = 1e-7

class Distribution(Layer):
    def __init__(self, dim, name='distribution', must_sample=False, **kwargs):
        self.dim = dim
        self.must_sample = must_sample

        kwargs = init_weights(self, **kwargs)
        kwargs = init_rngs(self, **kwargs)

        super(Distribution, self).__init__(name=name)

    def set_params(self):
        raise NotImplementedError()

    def get_params(self):
        raise NotImplementedError()

    def get_prob(self):
        raise NotImplementedError()

    def get(self):
        return self.get_prob(*self.get_params())

    def sample(self, n_samples):
        p = self.get_prob(*self.get_params())
        return self.f_sample(self.trng, p, size=(n_samples, self.dim)), theano.OrderedUpdates()

    def step_neg_log_prob(self, x, *params):
        p = self.get_prob(*params)
        return self.f_neg_log_prob(x, p)

    def neg_log_prob(self, x):
        p = self.get_prob(*self.get_params())
        return self.f_neg_log_prob(x, p)

    def entropy(self):
        p = self.get_prob(*self.get_params())
        return self.f_entropy(p)

    def kl_divergence(self, q):
        raise NotImplementedError()


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
        return [self.z]

    def get_prob(self, z):
        return T.nnet.sigmoid(z) * 0.9999 + 0.000005
        #return T.nnet.sigmoid(z)

class Gaussian(Distribution):
    def __init__(self, dim, name='gaussian', **kwargs):
        self.f_sample = _normal
        self.f_neg_log_prob = _neg_normal_log_prob
        self.f_entropy = _normal_entropy
        super(Gaussian, self).__init__(dim, name=name, **kwargs)

    def set_params(self):
        mu = np.zeros((self.dim,)).astype(floatX)
        log_sigma = np.zeros((self.dim,)).astype(floatX)

        self.params = OrderedDict(
            mu=mu, log_sigma=log_sigma)

    def get_params(self):
        return [self.mu, self.log_sigma]

    def get_prob(self, mu, log_sigma):
        return concatenate([mu, log_sigma])

    def step_kl_divergence(self, q, mu, log_sigma):
        mu_q = _slice(q, 0, self.dim)
        log_sigma_q = _slice(q, 1, self.dim)

        kl = log_sigma - log_sigma_q + 0.5 * (
            (T.exp(2 * log_sigma_q) + (mu - mu_q) ** 2) /
            T.exp(2 * log_sigma)
            - 1)
        return kl.sum(axis=kl.ndim-1)

    def kl_divergence(self, q):
        return self.step_kl_divergence(q, *self.get_params())


# BERNOULLI --------------------------------------------------------------------

def _binomial(trng, p, size=None):
    if size is None:
        size = p.shape
    return trng.binomial(p=p, size=size, n=1, dtype=p.dtype)

def _centered_binomial(trng, p, size=None):
    if size is None:
        size = p.shape
    return 2 * trng.binomial(p=0.5*(p+1), size=size, n=1, dtype=p.dtype) - 1.

def _cross_entropy(x, p):
    energy = -x * T.log(p) - (1 - x) * T.log(1 - p)
    #p = T.clip(p, _clip, 1.0 - _clip)
    #energy = T.nnet.binary_crossentropy(p, x)
    return energy.sum(axis=energy.ndim-1)

def _binary_entropy(p):
    entropy = -p * T.log(p) - (1 - p) * T.log(1 - p)
    #p_c = T.clip(p, _clip, 1.0 - _clip)
    #entropy = T.nnet.binary_crossentropy(p_c, p)
    return entropy.sum(axis=entropy.ndim-1)

# SOFTMAX ----------------------------------------------------------------------

def _softmax(x):
    axis = x.ndim - 1
    e_x = T.exp(x - x.max(axis=axis, keepdims=True))
    out = e_x / e_x.sum(axis=axis, keepdims=True)
    return out

def _sample_softmax(trng, p, size=None):
    if size is None:
        size = p.shape
    return trng.multinomial(pvals=p, size=size).astype('float32')

def _categorical_cross_entropy(x, p):
    p = T.clip(p, _clip, 1.0 - _clip)
    return T.nnet.binary_crossentropy(p, x).sum(axis=x.ndim-1)

def _categorical_entropy(p):
    p_c = T.clip(p, _clip, 1.0 - _clip)
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

def _neg_normal_log_prob(x, p):
    dim = p.shape[p.ndim-1] // 2
    mu = _slice(p, 0, dim)
    log_sigma = _slice(p, 1, dim)
    energy = 0.5 * (
        (x - mu)**2 / (T.exp(2 * log_sigma)) + 2 * log_sigma + T.log(2 * pi))
    return energy.sum(axis=energy.ndim-1)

def _normal_entropy(p):
    dim = p.shape[p.ndim-1] // 2
    log_sigma = _slice(p, 1, dim)
    entropy = 0.5 * T.log(2 * pi * e) + log_sigma
    return entropy.sum(axis=entropy.ndim-1)
