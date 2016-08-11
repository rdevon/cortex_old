'''Module for Binomial distributions.

'''

from collections import OrderedDict
import numpy as np
from theano import tensor as T

from . import Distribution
from ...utils import floatX


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


class Binomial(Distribution):
    '''Binomial distribution.

    '''
    _act_slope = 1 - 1e-4
    _act_incpt = 5e-6

    def __init__(self, dim, name='binomial', **kwargs):
        self.f_sample = _binomial
        self.f_neg_log_prob = _cross_entropy
        self.f_entropy = _binary_entropy
        super(Binomial, self).__init__(dim, name=name, **kwargs)

    def _act(self, X, as_numpy=False):
        if as_numpy:
            sigmoid = lambda x: 1. / (1. + np.exp(-x))
        else:
            sigmoid = T.nnet.sigmoid
        return sigmoid(X) * self._act_slope + self._act_incpt

    def init_params(self):
        z = np.zeros((self.dim,)).astype(floatX)
        self.params = OrderedDict(z=z)

    def quantile(self, epsilon, P):
        return (epsilon <= P).astype(floatX)

    def random_variables(self, size):
        return self.trng.uniform(size, dtype=floatX)

    def permute(self):
        h0 = T.zeros((self.dim,)).astype(floatX)[None, :]
        h = T.eye(self.dim).astype(floatX)
        return OrderedDict(mean=h0, perm=h)

    def viz(self, P0, P=None):
        if P is None:
            P = self.get_prob(*self.get_params())
        P0 = T.addbroadcast(P0, 0)
        return P - P0

    def get_energy_bias(self, x, z):
        return T.dot(x, z)


class CenteredBinomial(Binomial):
    '''Centered binomial.

    '''
    _distribution = 'centered_binomial'

    def _act(self, X, as_numpy=False):
        if as_numpy:
            Te = np
        else:
            Te = T
        return Te.tanh(X)

    def quantile(self, epsilon, P):
        return (2.0 * (epsilon <= P).astype(floatX) - 1.0)

    def neg_log_prob(self, X, P=None, sum_probs=True):
        if P is None: P = self.get_prob(*self.get_params())
        X = 0.5 * (x + 1.0)
        P = (0.5 * (p + 1.0)) * 0.9999 + 0.000005
        return self.f_neg_log_prob(X, P, sum_probs=sum_probs)


class ContinuousBinomial(Binomial):
    '''Continuous binomial.

    Note:
        Doesn't sample.

    '''
    _distribution = 'continuous_binomial'
    is_continuous = True

    def sample(self, n_samples, p=None):
        if p is None:
            p = self.get_prob(*self.get_params())
        return T.shape_padleft(p), theano.OrderedUpdates()


_classes = {'binomial': Binomial,
            'centered_binomial': CenteredBinomial,
            'continuous_binomial': ContinuousBinomial}