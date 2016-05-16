'''
Module for Theano probabilistic distributions.
'''

from collections import OrderedDict
import numpy as np
import theano
from theano import tensor as T

from . import Layer
from ..utils import e, floatX, pi
from ..utils.tools import (
    concatenate,
    init_rngs,
    init_weights,
    _slice
)


_clip = 1e-7 # clipping for Guassian distributions.

def resolve(c, conditional=False):
    '''Resolves Distribution subclass from str.'''
    resolve_dict = dict(
        binomial=Binomial,
        continuous_binomial=ContinuousBinomial,
        centered_binomial=CenteredBinomial,
        multinomial=Multinomial,
        gaussian=Gaussian,
        logistic=Logistic,
        laplace=Laplace
    )

    C = resolve_dict.get(c, None)
    if C is None:
        raise ValueError(C)
    if conditional:
        C = make_conditional(C)
    return C


class Distribution(Layer):
    '''Distribution parent class.

    Not meant to be used alone, use subclass.

    Attributes:
        has_kl: bool, convenience for if distribution subclass has exact KL.
        is_continuous: bool, whether distribution is continuous (as opposed to
            discrete).
        dim: int, dimension of distribution.
        must_sample: bool, whether sampling is required for calculating
            density.
        scale: int, scaling for distributions whose probs are higher order,
            such as Gaussian, which has mu and sigma.
        f_sample: function (optional), sampling function.
        f_neg_log_prob: function (optional), negative log probability funciton.
        f_entropy: function (optional), entropy function.
    '''
    has_kl = False
    is_continuous = False

    def __init__(self, dim, name='distribution', must_sample=False, scale=1,
                 **kwargs):
        '''Init function for Distribution class.

        Args:
            dim: int, dimension of distribution.
            must_sample: bool.
            scale: int, scale for distribution tesnor.
        '''
        self.dim = dim
        self.must_sample = must_sample
        self.scale = scale

        kwargs = init_weights(self, **kwargs)
        kwargs = init_rngs(self, **kwargs)

        super(Distribution, self).__init__(name=name)

    def set_params(self):
        raise NotImplementedError()

    def get_params(self):
        '''Fetches distribution parameters.'''
        raise NotImplementedError()

    def get_prob(self):
        '''Returns single tensory from params.'''
        raise NotImplementedError()

    def kl_divergence(self, q):
        '''KL divergence function.'''
        raise NotImplementedError()

    def __call__(self, z):
        '''Call function.'''
        raise NotImplementedError()

    def sample(self, n_samples, p=None):
        '''Samples from distribution.'''
        if p is None:
            p = self.get_prob(*self.get_params())
        if p.ndim == 1:
            size = (n_samples, p.shape[0] // self.scale)
        elif p.ndim == 2:
            size = (n_samples, p.shape[0], p.shape[1] // self.scale)
        elif p.ndim == 3:
            size = (n_samples, p.shape[0], p.shape[1], p.shape[2] // self.scale)
        elif p.ndim == 4:
            raise NotImplementedError('%d dim sampling not supported yet' % p.ndim)

        return self.f_sample(self.trng, p, size=size), theano.OrderedUpdates()

    def step_neg_log_prob(self, x, *params):
        '''Step negative log probability for scan.'''
        p = self.get_prob(*params)
        return self.f_neg_log_prob(x, p)

    def neg_log_prob(self, x, p=None, sum_probs=True):
        '''Negative log probability.'''
        if p is None:
            p = self.get_prob(*self.get_params())
        return self.f_neg_log_prob(x, p, sum_probs=sum_probs)

    def entropy(self, p=None):
        '''Entropy function.'''
        if p is None:
            p = self.get_prob(*self.get_params())
        return self.f_entropy(p)

    def get_center(self, p):
        '''Center of the distribution.'''
        return p

    def get_energy_bias(self, x, z):
        '''For use in RBMs and other energy based models'''
        raise NotImplementedError()

    def scale_for_energy_model(self, x, *params):
        '''Scales input for energy based models.'''
        return x


def make_conditional(C):
    '''Conditional distribution.

    Conditional distributions do not own their parameters, they are given,
    such as from an MLP.

    Args:
        C: Distribution subclass.
    Returns:
        Conditional subclass.
    '''
    class Conditional(C):
        def set_params(self): self.params = OrderedDict()
        def get_params(self): return []
    Conditional.__name__ = Conditional.__name__ + '_' + C.__name__

    return Conditional


class Binomial(Distribution):
    '''Binomial distribution.'''
    def __init__(self, dim, name='binomial', **kwargs):
        self.f_sample = _binomial
        self.f_neg_log_prob = _cross_entropy
        self.f_entropy = _binary_entropy
        super(Binomial, self).__init__(dim, name=name, **kwargs)

    def set_params(self):
        z = np.zeros((self.dim,)).astype(floatX)
        self.params = OrderedDict(z=z)

    def get_params(self):
        return [self.z]

    def get_prob(self, z):
        return T.nnet.sigmoid(z) * 0.9999 + 0.000005

    def split_prob(self, p):
        return p

    def __call__(self, z):
        return T.nnet.sigmoid(z) * 0.9999 + 0.000005

    def step_sample(self, epsilon, p):
        return (epsilon <= p).astype(floatX)

    def prototype_samples(self, size):
        return self.trng.uniform(size, dtype=floatX)

    def generate_latent_pair(self):
        h0 = T.zeros((self.dim,)).astype(floatX)[None, :]
        h = T.eye(self.dim).astype(floatX)
        return h0, h

    def visualize(self, p0, p=None):
        if p is None:
            p = self.get_prob(*self.get_params())
        p0 = T.addbroadcast(p0, 0)
        return p - p0

    def get_energy_bias(self, x, z):
        '''For use in RBMs and other energy based models'''
        return T.dot(x, z)


class CenteredBinomial(Binomial):
    '''Centered binomial.'''
    def get_prob(self, z):
        return T.nnet.sigmoid(2.0 * z) * 0.9999 + 0.000005

    def __call__(self, z):
        return T.nnet.sigmoid(2.0 * z) * 0.9999 + 0.000005

    def step_sample(self, epsilon, p):
        return (2.0 * (epsilon <= p).astype(floatX) - 1.0)

    def sample(self, n_samples, p=None):
        '''Samples from distribution.'''
        if p is None:
            p = self.get_prob(*self.get_params())
        if p.ndim == 1:
            size = (n_samples, p.shape[0] // self.scale)
        elif p.ndim == 2:
            size = (n_samples, p.shape[0], p.shape[1] // self.scale)
        elif p.ndim == 3:
            size = (n_samples, p.shape[0], p.shape[2], p.shape[3] // self.scale)
        elif p.ndim == 4:
            raise NotImplementedError('%d dim sampling not supported yet' % p.ndim)

        return (2.0 * self.f_sample(self.trng, p, size=size) - 1), theano.OrderedUpdates()

    def neg_log_prob(self, x, p=None, sum_probs=True):
        if p is None:
            p = self.get_prob(*self.get_params())
        x = 0.5 * (x + 1.0)
        return self.f_neg_log_prob(x, p, sum_probs=sum_probs)


class ContinuousBinomial(Binomial):
    '''Continuous binomial.

    Doesn't sample.
    '''
    def sample(self, n_samples, p=None):
        if p is None:
            p = self.get_prob(*self.get_params())
        return T.shape_padleft(p), theano.OrderedUpdates()


class Multinomial(Distribution):
    '''Multinomial distribuion.'''
    def __init__(self, dim, name='multinomial', **kwargs):
        self.f_sample = _sample_multinomial
        self.f_neg_log_prob = _categorical_cross_entropy
        self.f_entropy = _categorical_entropy
        super(Multinomial, self).__init__(dim, name=name, **kwargs)

    def set_params(self):
        z = np.zeros((self.dim,)).astype(floatX)
        self.params = OrderedDict(z=z)

    def get_prob(self, z):
        return _softmax(z)

    def __call__(self, z):
        return _softmax(z)


class Gaussian(Distribution):
    '''Gaussian distribution.'''
    has_kl = True
    is_continuous = True

    def __init__(self, dim, name='gaussian', clip=-10, **kwargs):
        self.f_sample = _normal
        self.f_neg_log_prob = _neg_normal_log_prob
        self.f_entropy = _normal_entropy
        self.clip = clip
        super(Gaussian, self).__init__(dim, name=name, scale=2, **kwargs)

    def set_params(self):
        mu = np.zeros((self.dim,)).astype(floatX)
        log_sigma = np.zeros((self.dim,)).astype(floatX)

        self.params = OrderedDict(
            mu=mu, log_sigma=log_sigma)

    def get_params(self):
        return [self.mu, self.log_sigma]

    def get_prob(self, mu, log_sigma):
        return concatenate([mu, log_sigma], axis=mu.ndim-1)

    def __call__(self, z):
        return z

    def get_center(self, p):
        mu = _slice(p, 0, p.shape[p.ndim-1] // self.scale)
        return mu

    def split_prob(self, p):
        mu        = _slice(p, 0, p.shape[p.ndim-1] // self.scale)
        log_sigma = _slice(p, 1, p.shape[p.ndim-1] // self.scale)
        return mu, log_sigma

    def step_kl_divergence(self, q, mu, log_sigma):
        mu_q = _slice(q, 0, self.dim)
        log_sigma_q = _slice(q, 1, self.dim)
        log_sigma_q = T.maximum(log_sigma_q, self.clip)
        log_sigma = T.maximum(log_sigma, self.clip)

        kl = log_sigma - log_sigma_q + 0.5 * (
            (T.exp(2 * log_sigma_q) + (mu - mu_q) ** 2) /
            T.exp(2 * log_sigma)
            - 1)
        return kl.sum(axis=kl.ndim-1)

    def kl_divergence(self, q):
        return self.step_kl_divergence(q, *self.get_params())

    def step_sample(self, epsilon, p):
        dim = p.shape[p.ndim-1] // self.scale
        mu = _slice(p, 0, dim)
        log_sigma = _slice(p, 1, dim)
        return mu + epsilon * T.exp(log_sigma)

    def prototype_samples(self, size):
        return self.trng.normal(
            avg=0, std=1.0,
            size=size,
            dtype=floatX
        )

    def step_neg_log_prob(self, x, *params):
        p = self.get_prob(*params)
        return self.f_neg_log_prob(x, p=p, clip=self.clip)

    def neg_log_prob(self, x, p=None, sum_probs=True):
        if p is None:
            p = self.get_prob(*self.get_params())
        return self.f_neg_log_prob(x, p, clip=self.clip, sum_probs=sum_probs)

    def standard_prob(self, x, p=None):
        if p is None:
            p = self.get_prob(*self.get_params())
        return T.exp(-self.neg_log_prob(x, p))

    def entropy(self, p=None):
        if p is None:
            p = self.get_prob(*self.get_params())
        return self.f_entropy(p, clip=self.clip)

    def generate_latent_pair(self):
        h0 = self.mu
        sigma = T.nlinalg.AllocDiag()(T.exp(self.log_sigma)).astype(floatX)
        h = 2 * sigma + h0[None, :]
        return h0, h

    def visualize(self, p0, p=None):
        if p is None:
            p = self.get_prob(*self.get_params())

        outs0 = self.split_prob(p0)
        outs = self.split_prob(p)
        y0_mu, y0_logsigma = outs0
        y_mu, y_logsigma = outs
        py = (y_mu - y0_mu) / T.exp(y0_logsigma)
        return py

    def scale_for_energy_model(self, x, mu, log_sigma):
        '''Scales input for energy based models.'''
        return x / T.exp(2 * log_sigma)

    def get_energy_bias(self, x, mu, log_sigma):
        '''For use in RBMs and other energy based models'''
        return -((x - mu) ** 2 / (2. * T.exp(log_sigma)) ** 2).sum(axis=x.ndim-1)


class Logistic(Distribution):
    '''Logistic distribution.

    Not to be confused with logistic function.
    '''
    is_continuous = True

    def __init__(self, dim, name='logistic', **kwargs):
        self.f_sample = _logistic
        self.f_neg_log_prob = _neg_logistic_log_prob
        self.f_entropy = _logistic_entropy
        super(Logistic, self).__init__(dim, name=name, scale=2, **kwargs)

    def set_params(self):
        mu = np.zeros((self.dim,)).astype(floatX)
        log_s = np.zeros((self.dim,)).astype(floatX)

        self.params = OrderedDict(
            mu=mu, log_s=log_s)

    def get_params(self):
        return [self.mu, self.log_s]

    def get_prob(self, mu, log_s):
        return concatenate([mu, log_s], axis=mu.ndim-1)

    def __call__(self, z):
        return z

    def get_center(self, p):
        mu = _slice(p, 0, p.shape[p.ndim-1] // self.scale)
        return mu

    def split_prob(self, p):
        mu    = _slice(p, 0, p.shape[p.ndim-1] // self.scale)
        log_s = _slice(p, 1, p.shape[p.ndim-1] // self.scale)
        return mu, log_s

    def step_sample(self, epsilon, p):
        dim = p.shape[p.ndim-1] // self.scale
        mu = _slice(p, 0, dim)
        log_s = _slice(p, 1, dim)
        return mu + T.log(epsilon / (1 - epsilon)) * T.exp(log_s)

    def prototype_samples(self, size):
        return self.trng.uniform(size=size, dtype=floatX)

    def step_neg_log_prob(self, x, *params):
        p = self.get_prob(*params)
        return self.f_neg_log_prob(x, p=p)

    def neg_log_prob(self, x, p=None, sum_probs=True):
        if p is None:
            p = self.get_prob(*self.get_params())
        return self.f_neg_log_prob(x, p, sum_probs=sum_probs)

    def standard_prob(self, x, p=None):
        if p is None:
            p = self.get_prob(*self.get_params())
        return T.exp(-self.neg_log_prob(x, p))

    def entropy(self, p=None):
        if p is None:
            p = self.get_prob(*self.get_params())
        return self.f_entropy(p)

    def generate_latent_pair(self):
        h0 = self.mu
        s = T.nlinalg.AllocDiag()(T.exp(self.log_s)).astype(floatX)
        h = 2 * s + h0[None, :]
        return h0, h

    def visualize(self, p0, p=None):
        if p is None:
            p = self.get_prob(*self.get_params())

        outs0 = self.split_prob(p0)
        outs = self.split_prob(p)
        y0_mu, y0_logs = outs0
        y_mu, y_logs = outs
        py = (y_mu - y0_mu) / T.exp(y0_logs)
        return py


class Laplace(Distribution):
    '''Laplace distribution.
    '''
    is_continuous = True

    def __init__(self, dim, name='laplace', **kwargs):
        self.f_sample = _laplace
        self.f_neg_log_prob = _neg_laplace_log_prob
        self.f_entropy = _laplace_entropy
        super(Laplace, self).__init__(dim, name=name, scale=2, **kwargs)

    def set_params(self):
        mu = np.zeros((self.dim,)).astype(floatX)
        log_b = np.zeros((self.dim,)).astype(floatX)

        self.params = OrderedDict(
            mu=mu, log_b=log_b)

    def get_params(self):
        return [self.mu, self.log_b]

    def get_prob(self, mu, log_b):
        return concatenate([mu, log_b], axis=mu.ndim-1)

    def __call__(self, z):
        return z

    def get_center(self, p):
        mu = _slice(p, 0, p.shape[p.ndim-1] // self.scale)
        return mu

    def split_prob(self, p):
        mu    = _slice(p, 0, p.shape[p.ndim-1] // self.scale)
        log_s = _slice(p, 1, p.shape[p.ndim-1] // self.scale)
        return mu, log_s

    def step_sample(self, epsilon, p):
        dim = p.shape[p.ndim-1] // self.scale
        mu = _slice(p, 0, dim)
        log_b = _slice(p, 1, dim)
        return mu + T.exp(log_b) * T.sgn(epsilon) * T.log(1.0 - 2 * abs(epsilon))

    def prototype_samples(self, size):
        return self.trng.uniform(size=size, dtype=floatX) - 0.5

    def step_neg_log_prob(self, x, *params):
        p = self.get_prob(*params)
        return self.f_neg_log_prob(x, p=p)

    def neg_log_prob(self, x, p=None, sum_probs=True):
        if p is None:
            p = self.get_prob(*self.get_params())
        return self.f_neg_log_prob(x, p, sum_probs=sum_probs)

    def standard_prob(self, x, p=None):
        if p is None:
            p = self.get_prob(*self.get_params())
        return T.exp(-self.neg_log_prob(x, p))

    def entropy(self, p=None):
        if p is None:
            p = self.get_prob(*self.get_params())
        return self.f_entropy(p)

    def generate_latent_pair(self):
        h0 = self.mu
        b = T.nlinalg.AllocDiag()(T.exp(self.log_b)).astype(floatX)
        h = 2 * b + h0[None, :]
        return h0, h

    def visualize(self, p0, p=None):
        if p is None:
            p = self.get_prob(*self.get_params())

        outs0 = self.split_prob(p0)
        outs = self.split_prob(p)
        y0_mu, y0_logs = outs0
        y_mu, y_logs = outs
        py = (y_mu - y0_mu) / T.exp(y0_logs)
        return py

# Various functions for distributions.
# BERNOULLI --------------------------------------------------------------------

def _binomial(trng, p, size=None):
    if size is None:
        size = p.shape
    return trng.binomial(p=p, size=size, n=1, dtype=p.dtype)

def _centered_binomial(trng, p, size=None):
    if size is None:
        size = p.shape
    return 2 * trng.binomial(p=0.5*(p+1), size=size, n=1, dtype=p.dtype) - 1.

def _cross_entropy(x, p, sum_probs=True):
    energy = -x * T.log(p) - (1 - x) * T.log(1 - p)
    if sum_probs:
        return energy.sum(axis=energy.ndim-1)
    else:
        return energy

def _binary_entropy(p):
    entropy = -p * T.log(p) - (1 - p) * T.log(1 - p)
    return entropy.sum(axis=entropy.ndim-1)

# SOFTMAX ----------------------------------------------------------------------

def _softmax(x):
    axis = x.ndim - 1
    e_x = T.exp(x - x.max(axis=axis, keepdims=True))
    out = e_x / e_x.sum(axis=axis, keepdims=True)
    return out

def _sample_multinomial(trng, p, size=None):
    if size is None:
        size = p.shape
    return trng.multinomial(pvals=p, size=size).astype('float32')

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

# GAUSSIAN ---------------------------------------------------------------------

def _normal(trng, p, size=None):
    dim = p.shape[p.ndim-1] // 2
    mu = _slice(p, 0, dim)
    log_sigma = _slice(p, 1, dim)

    if size is None:
        size = mu.shape
    return trng.normal(avg=mu, std=T.exp(log_sigma), size=size, dtype=floatX)

def _normal_prob(p):
    dim = p.shape[p.ndim-1] // 2
    mu = _slice(p, 0, dim)
    return mu

def _neg_normal_log_prob(x, p, clip=None, sum_probs=True):
    dim = p.shape[p.ndim-1] // 2
    mu = _slice(p, 0, dim)
    log_sigma = _slice(p, 1, dim)
    if clip is not None:
        log_sigma = T.maximum(log_sigma, clip)
    energy = 0.5 * (
        (x - mu)**2 / (T.exp(2 * log_sigma)) + 2 * log_sigma + T.log(2 * pi))
    if sum_probs:
        return energy.sum(axis=energy.ndim-1)
    else:
        return energy

def _normal_entropy(p, clip=None):
    dim = p.shape[p.ndim-1] // 2
    log_sigma = _slice(p, 1, dim)
    if clip is not None:
        log_sigma = T.maximum(log_sigma, clip)
    entropy = 0.5 * T.log(2 * pi * e) + log_sigma
    return entropy.sum(axis=entropy.ndim-1)

# LOGISTIC ---------------------------------------------------------------------

def _logistic(trng, p, size=None):
    dim = p.shape[p.ndim-1] // 2
    mu = _slice(p, 0, dim)
    log_s = _slice(p, 1, dim)
    if size is None:
        size = mu.shape
    epsilon = trng.uniform(size=size, dtype=floatX)
    return mu + T.log(epsilon / (1 - epsilon)) * T.exp(log_s)

def _neg_logistic_log_prob(x, p, sum_probs=True):
    dim = p.shape[p.ndim-1] // 2
    mu = _slice(p, 0, dim)
    log_s = _slice(p, 1, dim)
    energy = -(x - mu) / T.exp(log_s) + log_s + 2 * T.log(1 + T.exp((x - mu) / T.exp(log_s)))
    if sum_probs:
        return energy.sum(axis=energy.ndim-1)
    else:
        return energy

def _logistic_entropy(p):
    dim = p.shape[p.ndim-1] // 2
    log_s = _slice(p, 1, dim)
    entropy = log_s + 2.0
    return entropy.sum(axis=entropy.ndim-1)

# Laplace ---------------------------------------------------------------------

def _laplace(trng, p, size=None):
    dim = p.shape[p.ndim-1] // 2
    mu = _slice(p, 0, dim)
    log_b = _slice(p, 1, dim)
    if size is None:
        size = mu.shape
    epsilon = trng.uniform(size=size, dtype=floatX) - 0.5
    return mu + T.exp(log_b) * T.sgn(epsilon) * T.log(1.0 - 2 * abs(epsilon))

def _neg_laplace_log_prob(x, p, sum_probs=True):
    dim = p.shape[p.ndim-1] // 2
    mu = _slice(p, 0, dim)
    log_b = _slice(p, 1, dim)
    energy = T.log(2.0) + log_b + abs(x - mu) / T.exp(log_b)
    if sum_probs:
        return energy.sum(axis=energy.ndim-1)
    else:
        return energy

def _laplace_entropy(p):
    dim = p.shape[p.ndim-1] // 2
    log_b = _slice(p, 1, dim)
    entropy = log_b + T.log(2.) + 1.0
    return entropy.sum(axis=entropy.ndim-1)
