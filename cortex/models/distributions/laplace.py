'''Module for Laplace distribution.

'''

from collections import OrderedDict
import numpy as np
from theano import tensor as T

from . import Distribution
from ... import utils
from ...utils import floatX


def _laplace(trng, p, size=None):
    dim = p.shape[p.ndim-1] // 2
    mu = utils.slice(p, 0, dim)
    log_b = utils.slice(p, 1, dim)
    if size is None:
        size = mu.shape
    epsilon = trng.uniform(size=size, dtype=floatX) - 0.5
    return mu + T.exp(log_b) * T.sgn(epsilon) * T.log(1.0 - 2 * abs(epsilon))

def _neg_laplace_log_prob(x, p, sum_probs=True):
    dim = p.shape[p.ndim-1] // 2
    mu = utils.slice(p, 0, dim)
    log_b = utils.slice(p, 1, dim)
    energy = T.log(2.0) + log_b + abs(x - mu) / T.exp(log_b)
    if sum_probs:
        return energy.sum(axis=energy.ndim-1)
    else:
        return energy

def _laplace_entropy(p):
    dim = p.shape[p.ndim-1] // 2
    log_b = utils.slice(p, 1, dim)
    entropy = log_b + T.log(2.) + 1.0
    return entropy.sum(axis=entropy.ndim-1)


class Laplace(Distribution):
    '''Laplace distribution.

    :math:`p(x) = \\frac{1}{2 b} e^{-\\frac{|x - \mu|}{b}}`.

    '''
    is_continuous = True
    scale = 2

    def __init__(self, dim, name='laplace', **kwargs):
        self.f_sample = _laplace
        self.f_neg_log_prob = _neg_laplace_log_prob
        self.f_entropy = _laplace_entropy
        super(Laplace, self).__init__(dim, name=name, **kwargs)

    def init_params(self):
        mu = np.zeros((self.dim,)).astype(floatX)
        log_b = np.zeros((self.dim,)).astype(floatX)

        self.params = OrderedDict(
            mu=mu, log_b=log_b)

    def get_params(self):
        return [self.mu, self.log_b]

    def quantile(self, epsilon, P):
        mu, log_b = self.split_prob(P)
        return mu + T.exp(log_b) * T.sgn(epsilon) * T.log(1.0 - 2 * abs(epsilon))

    def random_variables(self, size):
        return self.trng.uniform(size=size, dtype=floatX) - 0.5

    def step_neg_log_prob(self, X, *params):
        P = self.get_prob(*params)
        return self.f_neg_log_prob(X, P)

    def neg_log_prob(self, X, P=None, sum_probs=True):
        if P is None:
            P = self.get_prob(*self.get_params())
        return self.f_neg_log_prob(X, P, sum_probs=sum_probs)

    def standard_prob(self, X, P=None):
        if P is None:
            P = self.get_prob(*self.get_params())
        return T.exp(-self.neg_log_prob(X, P))

    def entropy(self, P=None):
        if P is None:
            P = self.get_prob(*self.get_params())
        return self.f_entropy(P)

    def permute(self, scale=2.):
        h0 = self.mu
        b = T.nlinalg.AllocDiag()(T.exp(self.log_b)).astype(floatX)
        h = scale * b + h0[None, :]
        return OrderedDict(mean=h0, perm=h)

    def viz(self, p0, p=None):
        if p is None: p = self.get_prob(*self.get_params())

        outs0 = self.split_prob(p0)
        outs = self.split_prob(p)
        y0_mu, y0_logs = outs0
        y_mu, y_logs = outs
        py = (y_mu - y0_mu) / T.exp(y0_logs)
        return py


_classes = {'laplace': Laplace}