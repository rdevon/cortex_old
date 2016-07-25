'''
Module for Theano probabilistic distributions.
'''

from collections import OrderedDict
import numpy as np
import theano
from theano import tensor as T

from . import init_rngs, Cell
from ..utils import concatenate, e, floatX, pi, _slice, _slice2
from ..utils.tools import _p


_clip = 1e-7 # clipping for Guassian distributions.


class Distribution(Cell):
    '''Distribution parent class.

    Not meant to be used alone, use subclass.

    Attributes:
        has_kl (bool): convenience for if distribution subclass has exact KL.
        is_continuous (bool): whether distribution is continuous (as opposed to
            discrete).
        dim (int): dimension of distribution.
        must_sample (bool): whether sampling is required for calculating
            density.
        scale (int): scaling for distributions whose probs are higher order,
            such as Gaussian, which has mu and sigma.
        f_sample (Optional[function]): sampling function.
        f_neg_log_prob (Optional[function]): negative log probability funciton.
        f_entropy (Optional[function]): entropy function.

    '''
    _args = ['dim']
    _required = ['dim']
    _dist_map = {'input': 'cell_type', 'output': 'cell_type', 'P': 'cell_type',
                 'samples': 'cell_type'}
    _dim_map = {'input': 'dim', 'output': 'dim', 'P': 'dim', 'samples': 'dim'}
    _costs = {
        'nll': '_cost',
        'negative_log_likelihood': '_cost',
        'kl_divergence': 'kl_divergence'
    }

    has_kl = False
    is_continuous = False
    scale = 1
    must_sample = False
    base_distribution = None

    def __init__(self, dim, name='distribution_proto', **kwargs):
        '''Init function for Distribution class.

        Args:
            dim (int): dimension of distribution.

        '''
        self.dim = dim
        super(Distribution, self).__init__(name=name, **kwargs)

    def _act(self, X, as_numpy=False):
        return X

    @classmethod
    def set_link_value(C, key, dim=None, **kwargs):
        if key not in ['input', 'output']:
            return super(Distribution, C).set_link_value(key, dim=dim, **kwargs)

        if dim is not None:
            return C.scale * dim
        else:
            raise ValueError

    @classmethod
    def get_link_value(C, link, key):
        if key not in ['input', 'output']:
            return super(Distribution, C).get_link_value(link, key)
        if link.value is None:
            raise ValueError
        if key == 'output':
            return ('dim', link.value)
        else:
            return ('dim', link.value / C.scale)

    @classmethod
    def factory(C, cell_type=None, conditional=False, **kwargs):
        '''Resolves Distribution subclass from str.

        Args:
            layer_type (str): string id for distribution class.
            conditional (Optional[bool]): if True, then use `Conditional` class.

        Returns:
            Distribution.

        '''
        reqs = OrderedDict(
            (k, v) for k, v in kwargs.iteritems() if k in C._required)
        options = dict((k, v) for k, v in kwargs.iteritems() if not k in C._required)

        for req in C._required:
            if req not in reqs.keys():
                raise TypeError('Required argument %s not provided for '
                                'constructor of %s' % (req, C))

        if conditional:
            C = _conditionals[C.__name__]

        return C(*reqs.values(), **options)

    def init_params(self):
        raise NotImplementedError()

    def get_params(self):
        '''Fetches distribution parameters.

        '''
        raise NotImplementedError()

    def get_prob(self, *args):
        '''Returns single tensory from params.

        '''
        return self._act(concatenate(args))

    def split_prob(self, p):
        '''Slices single tensor into constituent parts.

        '''
        slice_size = p.shape[p.ndim-1] // self.scale
        slices = [_slice2(p, i * slice_size, (i + 1) * slice_size)
                  for i in range(self.scale)]
        return slices

    def get_center(self, p):
        '''Gets center of distribution.

        Note ::
            Assumed to be first parameter.

        '''
        slices = self.split_prob(p)
        return slices[0]

    def _feed(self, *args):
        return self._act(concatenate(args))

    def generate_random_variables(self, shape, P=None):
        if P is None:
            P = self.get_prob(*self.get_params())
        if P.ndim == 0:
            P = T.zeros((self.dim,)).astype(floatX) + P
        if P.ndim == 1:
            shape = shape + (P.shape[0] // self.scale,)
        elif P.ndim == 2:
            shape = shape + (P.shape[0], P.shape[1] // self.scale)
        elif P.ndim == 3:
            shape = shape + (P.shape[0], P.shape[1], P.shape[2] // self.scale)
        elif P.ndim == 4:
            shape = shape + (P.shape[0], P.shape[1], P.shape[2],
                             P.shape[3] // self.scale)
        else:
            raise ValueError(P.ndim)
        return self.random_variables(shape)

    def _sample(self, epsilon, P=None):
        '''Samples from distribution.

        '''
        if P is None:
            P = self.get_prob(*self.get_params())
        return self.quantile(epsilon, P)

    def simple_sample(self, n_samples, P=None):
        epsilon = self.generate_random_variables((n_samples,), P=P)
        return self._sample(epslion, P=P)

    def _cost(self, X=None, P=None):
        if X is None:
            raise TypeError('X (ground truth) must be provided.')
        return self.neg_log_prob(X, P=P).mean()

    def step_neg_log_prob(self, X, *params):
        '''Step negative log probability for scan.

        Args:
            x (T.tensor): input.
            *params: theano shared variables.

        Returns:
            T.tensor: :math:`-\log p(x)`.

        '''
        P = self.get_prob(*params)
        return self.f_neg_log_prob(X, P)

    def neg_log_prob(self, X, P=None, sum_probs=True):
        '''Negative log probability.

        Args:
            x (T.tensor): input.
            p (Optional[T.tensor]): probability.
            sum_probs (bool): whether to sum the last axis.

        Returns:
            T.tensor: :math:`-\log p(x)`.

        '''
        if P is None:
            P = self.get_prob(*self.get_params())
        return self.f_neg_log_prob(X, P, sum_probs=sum_probs)

    def entropy(self, P=None):
        '''Entropy function.

        Args:
            p (T.tensor): probability.

        Returns:
            T.tensor: entropy.

        '''
        if P is None:
            P = self.get_prob(*self.get_params())
        return self.f_entropy(P)

    def get_energy_bias(self, x, z):
        '''For use in RBMs and other energy based models.

        Args:
            x (T.tensor): input.
            z (T.tensor): distribution tensor.

        '''
        raise NotImplementedError()

    def scale_for_energy_model(self, x, *params):
        '''Scales input for energy based models.

        Args:
            x (T.tensor): input.

        '''
        return x


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

    def get_params(self):
        return [self.z]

    def quantile(self, epsilon, P):
        return (epsilon <= P).astype(floatX)

    def random_variables(self, size):
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
        if P is None:
            P = self.get_prob(*self.get_params())
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


class Logistic(Distribution):
    '''Logistic distribution.

    :math:`p(x)=\\frac{e^{\\frac{x - \mu}{s}}}{s(1+e^{\\frac{x - \mu}{s}})^2}`

    Note:
        Not to be confused with logistic function.

    '''
    is_continuous = True
    scale = 2

    def __init__(self, dim, name='logistic', **kwargs):
        self.f_sample = _logistic
        self.f_neg_log_prob = _neg_logistic_log_prob
        self.f_entropy = _logistic_entropy
        super(Logistic, self).__init__(dim, name=name, **kwargs)

    def init_params(self):
        mu = np.zeros((self.dim,)).astype(floatX)
        log_s = np.zeros((self.dim,)).astype(floatX)

        self.params = OrderedDict(
            mu=mu, log_s=log_s)

    def get_params(self):
        return [self.mu, self.log_s]

    def quantile(self, epsilon, P):
        mu, log_s = self.split_prob(P)
        return mu + T.log(epsilon / (1 - epsilon)) * T.exp(log_s)

    def random_variables(self, size):
        return self.trng.uniform(size=size, dtype=floatX)

    def step_neg_log_prob(self, X, *params):
        P = self.get_prob(*params)
        return self.f_neg_log_prob(X, P=P)

    def neg_log_prob(self, X, P=None, sum_probs=True):
        if P is None:
            P = self.get_prob(*self.get_params())
        return self.f_neg_log_prob(X, P, sum_probs=sum_probs)

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
    g = (x - mu) / T.exp(log_s)
    energy = -g + log_s + 2 * T.log(1 + T.exp(g))
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

_classes = {'binomial': Binomial, 'continuous_binomial': ContinuousBinomial,
            'centered_binomial': CenteredBinomial, 'multinomial': Multinomial,
            'gaussian': Gaussian, 'logistic': Logistic, 'laplace': Laplace}

_conditionals = {}

def make_conditional(C):
    '''Conditional distribution.

    Conditional distributions do not own their parameters, they are given,
    such as from an MLP.

    Args:
        C (Distribution).

    Returns:
        Conditional.

    '''
    if not issubclass(C, Distribution):
        raise TypeError('Conditional distribution not possible with %s' % C)

    class Conditional(C):
        base_distribution = C
        def init_params(self): self.params = OrderedDict()

        def get_params(self): return []

        def neg_log_prob(self, X, P, sum_probs=True):
            return super(Conditional, self).neg_log_prob(
                X, P=P, sum_probs=sum_probs)

        def _cost(self, X=None, P=None):
            if X is None:
                raise TypeError('X (ground truth) must be provided.')
            if P is None:
                session = self.manager.get_session()
                if _p(self.name, 'P') not in session.tensors.keys():
                    raise TypeError('%s.P not found in graph nor provided'
                                    % self.name)
                P = session.tensors[_p(self.name, 'P')]
            return self.neg_log_prob(X, P=P).mean()

        def generate_random_variables(self, shape, P=None):
            if P is None:
                raise TypeError('P (distribution) must be provided')

            return super(Conditional, self).generate_random_variables(
                shape, P=P)

        def _sample(self, epsilon, P=None):
            if P is None:
                raise TypeError('P (distribution) must be provided')

            return super(Conditional, self)._sample(epsilon, P=P)

    Conditional.__name__ = Conditional.__name__ + '_' + C.__name__

    return Conditional

keys = _classes.keys()
for k in keys:
    v = _classes[k]
    C = make_conditional(v)
    _conditionals[v.__name__] = C
    _classes['conditional_' + k] = C
