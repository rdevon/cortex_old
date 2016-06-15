'''Module for MLP model.

'''

from collections import OrderedDict
import numpy as np
import theano
from theano import tensor as T
import warnings

from .distributions import Distribution, resolve as resolve_distribution
from . import Layer
from ..utils import floatX
from ..utils.tools import (
    concatenate,
    init_rngs,
    init_weights,
    norm_weight
)


def resolve(c):
    '''Resolves the MLP subclass from str.

    Note:
        Currently, only one MLP supported. More in the future.

    '''
    if c == 'mlp' or c is None:
        return MLP
    else:
        raise ValueError(c)


class MLP(Layer):
    '''Multilayer perceptron model.

    Attributes:
        dim_in (int): input dimension.
        dim_out (int): output dimension.
        distribution (Optional[Distribution]): distribution of output. Used for
            sampling, density calculations, etc.
        dim_h (Optional[int]): dimension of hidden layer.
        dim_hs (Optional[list]): for multiple hidden layers.
        n_layers (int): number of output and hidden layers.
        h_act (str): hidden activation string.

    '''
    must_sample = False
    def __init__(self, dim_in, dim_out, dim_h=None, n_layers=None, dim_hs=None,
                 h_act='T.nnet.sigmoid', distribution='binomial',
                 distribution_args=None, name='MLP', **kwargs):
        '''Init function for MLP.

        Args:
            dim_in (int): input dimension.
            dim_out (int): output dimension.
            dim_h (Optional[int]): dimension of hidden layer.
            n_layers (int): number of output and hidden layers.
            dim_hs (Optional[list]): for multiple hidden layers.
            h_act (str): hidden activation string.
            distribution (Optional[Distribution]): distribution of output. Used for
                sampling, density calculations, etc.
            distribution_args (Optional[dict]): optional arguments for
                distribution.
            name (str): name of model.
            **kwargs: optional keyword arguments.

        '''

        if distribution_args is None: distribution_args = dict()
        self.dim_in = dim_in

        if isinstance(distribution, Distribution):
            self.distribution = distribution
        elif distribution is not None:
            self.distribution = resolve_distribution(
                distribution, conditional=True)(
                dim_out, **distribution_args)
        else:
            self.distribution = None

        self.dim_out = dim_out
        if self.distribution is not None:
            self.dim_out *= self.distribution.scale

        if dim_h is None:
            if dim_hs is None:
                dim_hs = []
            else:
                dim_hs = [dim_h for dim_h in dim_hs]
            assert n_layers is None
        else:
            assert dim_hs is None
            dim_hs = []
            for l in xrange(n_layers - 1):
                dim_hs.append(dim_h)
        dim_hs.append(self.dim_out)
        self.dim_hs = dim_hs
        self.n_layers = len(dim_hs)
        assert self.n_layers > 0

        self.h_act = h_act

        kwargs = init_weights(self, **kwargs)
        kwargs = init_rngs(self, **kwargs)
        super(MLP, self).__init__(name=name, **kwargs)

    @staticmethod
    def factory(dim_in=None, dim_out=None,
                **kwargs):
        '''MLP factory.

        Convenience function for building MLPs.

        Args:
            dim_in (int): input dimension.
            dim_out (int): output dimension.
            **kwargs: construction keyword arguments.

        '''
        if dim_in is None or dim_out is None:
            raise TypeError('Both dim in (%r) and dim_out (%r) must be set'
                            % (dim_in, dim_out))
        return MLP(dim_in, dim_out, **kwargs)

    def sample(self, p, n_samples=1):
        '''Sample from the conditional distribution.

        Args:
            p (T.tensor): probability.
            n_samples (int): number of samples.

        Returns:
            T.tensor: (samples).
            theano.OrderedUpdates: updates.

        '''
        assert self.distribution is not None
        return self.distribution.sample(n_samples, p=p)

    def neg_log_prob(self, x, p, sum_probs=True):
        '''Negative log probability.

        Args:
            x (T.tensor): sample.
            p (T.tensor): probability.
            sum_probs (bool): whether to sum the last axis in neg log prob.

        Returns:
            T.tensor: negative log probabilities.

        '''
        assert self.distribution is not None
        return self.distribution.neg_log_prob(x, p, sum_probs=sum_probs)

    def entropy(self, p):
        '''Entropy function.

        Args:
            p (T.tensor): probability.

        Returns:
            T.tensor: entropies.

        '''
        assert self.distribution is not None
        return self.distribution.entropy(p)

    def get_center(self, p):
        '''
        Args:
            p (T.tensor): distribution tensor.

        Returns:
            T.tensor: center of distribution.

        '''
        assert self.distribution is not None
        return self.distribution.get_center(p)

    def split_prob(self, p):
        '''
        Args:
            p (T.tensor): distribution tensor.

        Returns:
            T.tuple: split distribution.

        '''
        assert self.distribution is not None
        return self.distribution.split_prob(p)

    def set_params(self):
        self.params = OrderedDict()

        for l in xrange(self.n_layers):
            if l == 0:
                dim_in = self.dim_in
            else:
                dim_in = self.dim_hs[l-1]
            dim_out = self.dim_hs[l]

            W = norm_weight(dim_in, dim_out,
                            scale=self.weight_scale, ortho=False)
            b = np.zeros((dim_out,)).astype(floatX)

            self.params['W%d' % l] = W
            self.params['b%d' % l] = b

    def get_params(self):
        params = []
        for l in xrange(self.n_layers):
            W = self.__dict__['W%d' % l]
            b = self.__dict__['b%d' % l]
            params += [W, b]
        return params

    def get_decay_params(self):
        decay_keys = [k for k in self.params.keys() if 'b' not in k]
        decay_params = OrderedDict((self.name + '.' + k, self.__dict__[k])
            for k in decay_keys)
        return decay_params

    def step_call(self, x, *params):
        '''Step feed forward MLP.

        Args:
            x (T.tensor): input.
            *params: theano shared variables.

        Returns:
            OrderedDict: results at every layer.

        '''
        params = list(params)
        outs = OrderedDict(x=x)
        for l in xrange(self.n_layers):
            W = params.pop(0)
            b = params.pop(0)

            if self.weight_noise:
                self.logger.debug(
                    'Using weight noise in layer %d for MLP %s' % (l, self.name))
                W += self.trng.normal(avg=0., std=self.weight_noise, size=W.shape)

            preact = T.dot(x, W) + b

            if l < self.n_layers - 1:
                x = eval(self.h_act)(preact)
                outs['preact_%d' % l] = preact
                outs[l] = x
            else:
                if self.distribution is not None:
                    x = self.distribution(preact)
                else:
                    x = eval(self.h_act)(preact)
                outs['z'] = preact
                outs['p'] = x

            if self.dropout and l != self.n_layers - 1:
                self.logger.debug('Adding dropout to layer {layer} for MLP '
                                  '`{name}`'.format(layer=l, name=self.name))
                if self.h_act == 'T.tanh':
                    x_d = self.trng.binomial(x.shape, p=1-self.dropout, n=1,
                                             dtype=x.dtype)
                    x = 2. * (x_d * (x + 1.) / 2) / (1 - self.dropout) - 1
                elif self.h_act in ['T.nnet.sigmoid', 'T.nnet.softplus',
                                    'lambda x: x']:
                    x_d = self.trng.binomial(x.shape, p=1-self.dropout, n=1,
                                             dtype=x.dtype)
                    x = x * x_d / (1 - self.dropout)
                else:
                    raise NotImplementedError('No dropout for %s yet' % activ)

        assert len(params) == 0, params
        return outs

    def __call__(self, x):
        '''Call function.

        Args:
            x (T.tensor): input.

        Returns:
            OrderedDict: results at every layer.

        '''
        params = self.get_params()
        outs = self.step_call(x, *params)
        return outs

    def feed(self, x):
        '''Simple feed function.

        Args:
            x (T.tensor): input.

        Returns:
            T.tensor: output

        '''
        return self.__call__(x)['p']

    def step_feed(self, x, *params):
        '''Step feed function.

        Args:
            x (T.tensor): input.
            *params: theano shared variables.

        Returns:
            T.tensor: output

        '''
        return self.step_call(x, *params)['p']

    def preact(self, x):
        '''Simple feed preactivation function.

        Args:
            x (T.tensor): input.

        Returns:
            T.tensor: preactivation

        '''
        return self.__call__(x)['z']

    def step_preact(self, x, *params):
        '''Step feed preactivation function.

        Args:
            x (T.tensor): input.
            *params: theano shared variables.

        Returns:
            T.tensor: preactivation

        '''
        return self.step_call(x, *params)['z']
