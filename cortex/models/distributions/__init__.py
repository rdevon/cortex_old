'''
Module for Theano probabilistic distributions.
'''

from collections import OrderedDict
import numpy as np
import theano
from theano import tensor as T

from .. import init_rngs, Cell
from ...utils import concatenate, e, floatX, pi, _slice, _slice2
from ...utils.tools import _p


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
        if isinstance(P, float) or P.ndim == 0:
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
        return self._sample(epsilon, P=P)

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


_classes = {}
from . import binomial, gaussian, laplace, logistic, multinomial
_modules =[binomial, gaussian, laplace, logistic, multinomial]
for module in _modules: _classes.update(**module._classes)
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