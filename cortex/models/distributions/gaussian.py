'''Module for Gaussian distributions.

'''

from collections import OrderedDict
import numpy as np
from theano import tensor as T

from . import Distribution, _clip
from ... import utils
from ...utils import e, floatX, pi


def _normal(trng, p, size=None):
    dim = p.shape[p.ndim-1] // 2
    mu = utils.tslice(p, 0, dim)
    log_sigma = utils.slice(p, 1, dim)

    if size is None:
        size = mu.shape
    return trng.normal(avg=mu, std=T.exp(log_sigma), size=size, dtype=floatX)

def _normal_prob(p):
    dim = p.shape[p.ndim-1] // 2
    mu = utils.slice(p, 0, dim)
    return mu

def _neg_normal_log_prob(x, p, clip=None, sum_probs=True):
    dim = p.shape[p.ndim-1] // 2
    mu = utils.slice(p, 0, dim)
    log_sigma = utils.tslice(p, 1, dim)
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
    log_sigma = utils.slice(p, 1, dim)
    if clip is not None:
        log_sigma = T.maximum(log_sigma, clip)
    entropy = 0.5 * T.log(2 * pi * e) + log_sigma
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

    def get_params(self):
        return [self.mu, self.log_sigma]

    @staticmethod
    def kl_divergence(mu_p, log_sigma_p, mu_q, log_sigma_q):
        log_sigma = T.maximum(log_sigma_p, _clip)
        log_sigma_q = T.maximum(log_sigma_q, _clip)

        kl = log_sigma_q - log_sigma_p + 0.5 * (
            (T.exp(2 * log_sigma_p) + (mu_q - mu_p) ** 2) /
            T.exp(2 * log_sigma_q)
            - 1)
        return kl.sum(axis=kl.ndim-1)

    def quantile(self, epsilon, P):
        mu, log_sigma = self.split_prob(P)
        return mu + epsilon * T.exp(log_sigma)

    def random_variables(self, size):
        return self.trng.normal(avg=0, std=1.0, size=size, dtype=floatX)

    def step_neg_log_prob(self, X, *params):
        P = self.get_prob(*params)
        return self.f_neg_log_prob(X, P=P, clip=self.clip)

    def neg_log_prob(self, X, P=None, sum_probs=True):
        if P is None:
            P = self.get_prob(*self.get_params())
        return self.f_neg_log_prob(X, P, clip=self.clip, sum_probs=sum_probs)

    def standard_prob(self, X, P=None):
        if P is None:
            P = self.get_prob(*self.get_params())
        return T.exp(-self.neg_log_prob(X, P))

    def entropy(self, P=None):
        if P is None:
            P = self.get_prob(*self.get_params())
        return self.f_entropy(P, clip=self.clip)

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
        '''Scales input for energy based models.

        '''
        return x / T.exp(2 * log_sigma)

    def get_energy_bias(self, x, mu, log_sigma):
        '''For use in RBMs and other energy based models.

        '''
        return -((x - mu) ** 2 / (2. * T.exp(log_sigma)) ** 2).sum(axis=x.ndim-1)


_classes = {'gaussian': Gaussian}