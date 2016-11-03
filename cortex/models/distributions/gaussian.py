'''Module for Gaussian distributions.

'''

from collections import OrderedDict
import numpy as np
import random
from theano import tensor as T

from . import Distribution, _clip
from ... import utils
from ...utils import e, floatX, pi, scan


def _normal(trng, p, size=None):
    dim = p.shape[p.ndim-1] // 2
    mu = utils.tslice(p, 0, dim)
    log_sigma = utils.slice(p, 1, dim)

    if size is None: size = mu.shape
    return trng.normal(avg=mu, std=T.exp(log_sigma), size=size, dtype=floatX)

def _normal_prob(p):
    dim = p.shape[p.ndim-1] // 2
    mu = utils.slice(p, 0, dim)
    return mu

def _neg_normal_log_prob(x, p, clip=None, sum_probs=True):
    dim = p.shape[p.ndim-1] // 2
    mu = utils.slice(p, 0, dim)
    log_sigma = utils.tslice(p, 1, dim)
    if clip is not None: log_sigma = T.maximum(log_sigma, clip)
    energy = 0.5 * (
        (x - mu)**2 / (T.exp(2 * log_sigma)) + 2 * log_sigma + T.log(2 * pi))
    if sum_probs:
        return energy.sum(axis=energy.ndim-1)
    else:
        return energy

def _normal_entropy(p, clip=None):
    dim = p.shape[p.ndim-1] // 2
    log_sigma = utils.slice(p, 1, dim)
    if clip is not None: log_sigma = T.maximum(log_sigma, clip)
    entropy = 0.5 * T.log(2 * pi * e) + log_sigma
    return entropy.sum(axis=entropy.ndim-1)

def _normal_unit_variance(trng, mu, size=None):
    if size is None: size = mu.shape
    return trng.normal(avg=mu, std=T.ones_like(mu), size=size, dtype=floatX)

def _normal_prob_unit_variance(mu):
    return mu

def _neg_normal_log_prob_unit_variance(x, mu, clip=None, sum_probs=True):
    energy = 0.5 * (
        (x - mu)**2 + T.log(2 * pi))
    if sum_probs:
        return energy.sum(axis=energy.ndim-1)
    else:
        return energy

def _normal_entropy_unit_variance(mu, clip=None):
    entropy = 0.5 * T.log(2 * pi * e)
    return entropy.sum(axis=entropy.ndim-1)


class Gaussian(Distribution):
    '''Gaussian distribution.

    '''
    has_kl = True
    is_continuous = True
    scale = 2

    def __init__(self, dim, name='gaussian', clip=-10, **kwargs):
        self.f_sample = _normal
        self.f_neg_log_prob = _neg_normal_log_prob
        self.f_entropy = _normal_entropy
        self.clip = clip
        super(Gaussian, self).__init__(dim, name=name, **kwargs)

    def init_params(self):
        mu = np.zeros((self.dim,)).astype(floatX)
        log_sigma = np.zeros((self.dim,)).astype(floatX)

        self.params = OrderedDict(
            mu=mu, log_sigma=log_sigma)

    @staticmethod
    def kl_divergence(mu_p, log_sigma_p, mu_q, log_sigma_q):
        log_sigma = T.maximum(log_sigma_p, self.clip)
        log_sigma_q = T.maximum(log_sigma_q, self.clip)

        kl = log_sigma_q - log_sigma_p + 0.5 * (
            (T.exp(2 * log_sigma_p) + (mu_q - mu_p) ** 2) /
            T.exp(2 * log_sigma_q)
            - 1)
        return kl.sum(axis=kl.ndim-1)

    def step_neg_log_prob(self, X, *params):
        '''Step negative log probability for scan.

        Args:
            x (T.tensor): input.
            *params: theano shared variables.

        Returns:
            T.tensor: :math:`-\log p(x)`.

        '''
        P = self.get_prob(*params)
        return self.f_neg_log_prob(X, P, clip=self.clip)

    def neg_log_prob(self, X, P=None, sum_probs=True):
        '''Negative log probability.

        Args:
            x (T.tensor): input.
            p (Optional[T.tensor]): probability.
            sum_probs (bool): whether to sum the last axis.

        Returns:
            T.tensor: :math:`-\log p(x)`.

        '''
        if P is None: P = self.get_prob(*self.get_params())
        return self.f_neg_log_prob(X, P, sum_probs=sum_probs, clip=self.clip)

    def quantile(self, epsilon, P):
        mu, log_sigma = self.split_prob(P)
        return mu + epsilon * T.exp(log_sigma)

    def random_variables(self, size):
        return self.trng.normal(avg=0, std=1.0, size=size, dtype=floatX)

    def standard_prob(self, X, P=None):
        if P is None:
            P = self.get_prob(*self.get_params())
        return T.exp(-self.neg_log_prob(X, P))

    def permute(self, scale=2.):
        h0 = self.mu
        sigma = T.nlinalg.AllocDiag()(T.exp(self.log_sigma)).astype(floatX)
        h = scale * sigma + h0[None, :]
        return OrderedDict(mean=h0, perm=h)

    def viz(self, P0, P=None):
        if P is None: P = self.get_prob(*self.get_params())

        outs0 = self.split_prob(P0)
        outs = self.split_prob(P)
        y0_mu, y0_logsigma = outs0
        y_mu, y_logsigma = outs
        Py = (y_mu - y0_mu) / T.exp(y0_logsigma)
        return Py

    def scale_for_energy_model(self, x, mu, log_sigma):
        '''Scales input for energy based models.

        '''
        return x / T.exp(2 * log_sigma)

    def get_energy_bias(self, x, mu, log_sigma):
        '''For use in RBMs and other energy based models.

        '''
        return -((x - mu) ** 2 / (2. * T.exp(log_sigma)) ** 2).sum(axis=x.ndim-1)

    def grid2d(self, idx1=None, idx2=None, n_steps=10, std=2.0, random_idx=False):
        if random_idx:
            idx1 = random.randint(0, self.dim - 1)
            idx2 = random.randint(0, self.dim - 1)
        else:
            if idx1 is None or idx2 is None:
                raise TypeError('Both idx need to be set if not random')

        def step(x):
            i = x % n_steps
            j = x // n_steps
            vec = T.zeros((self.dim, ))
            vec = T.set_subtensor(vec[idx1], (i - n_steps / 2.) * std / float(n_steps))
            vec = T.set_subtensor(vec[idx2], (j - n_steps / 2.) * std / float(n_steps))
            return vec

        a = T.arange(n_steps ** 2)
        b, _ = scan(step, [a], [None], [], a.shape[0])
        b = b.reshape((n_steps, n_steps, self.dim)).astype(floatX)
        return b


class GaussianUnitVariance(Distribution):
    '''Gaussian distribution.

    '''
    has_kl = True
    is_continuous = True
    scale = 1

    def __init__(self, dim, name='gaussian', clip=-10, **kwargs):
        self.f_sample = _normal_unit_variance
        self.f_neg_log_prob = _neg_normal_log_prob_unit_variance
        self.f_entropy = _normal_entropy_unit_variance
        self.clip = clip
        super(GaussianUnitVariance, self).__init__(dim, name=name, **kwargs)

    def init_params(self):
        mu = np.zeros((self.dim,)).astype(floatX)

        self.params = OrderedDict(mu=mu)

    @staticmethod
    def kl_divergence(mu_p, mu_q, log_sigma_q):
        return Gaussian.kl_divergence(
            mu_p, T.zeros_like(mu_p), mu_q, log_sigma_q)

    def quantile(self, epsilon, P):
        return P + epsilon

    def random_variables(self, size):
        return self.trng.normal(avg=0, std=1.0, size=size, dtype=floatX)

    def permute(self, scale=2.):
        h0 = self.mu
        sigma = T.nlinalg.AllocDiag()(T.ones_like(h0)).astype(floatX)
        h = scale * sigma + h0[None, :]
        return OrderedDict(mean=h0, perm=h)

    def viz(self, P0, P=None):
        if P is None: P = self.get_prob(*self.get_params())

        y0_mu = self.split_prob(P0)
        y_mu = self.split_prob(P)
        Py = y_mu - y0_mu
        return Py

    def scale_for_energy_model(self, x, mu):
        '''Scales input for energy based models.

        '''
        return x

    def get_energy_bias(self, x, mu):
        '''For use in RBMs and other energy based models.

        '''
        return -((x - mu) ** 2 / 2. ** 2).sum(axis=x.ndim-1)

    def grid2d(self, idx1=None, idx2=None, n_steps=10, std=2.0, random_idx=False):
        if random_idx:
            idx1 = random.randint(0, self.dim - 1)
            idx2 = random.randint(0, self.dim - 1)
        else:
            if idx1 is None or idx2 is None:
                raise TypeError('Both idx need to be set if not random')

        def step(x):
            i = x % n_steps
            j = x // n_steps
            vec = T.zeros((self.dim, ))
            vec = T.set_subtensor(vec[idx1], (i - n_steps / 2.) * std / float(n_steps))
            vec = T.set_subtensor(vec[idx2], (j - n_steps / 2.) * std / float(n_steps))
            return vec

        a = T.arange(n_steps ** 2)
        b, _ = scan(step, [a], [None], [], a.shape[0])
        b = b.reshape((n_steps, n_steps, self.dim)).astype(floatX)
        return b


_classes = {'gaussian': Gaussian, 'gaussian_unit_variance': GaussianUnitVariance}