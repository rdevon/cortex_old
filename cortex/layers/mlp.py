'''Module for MLP model.

'''

from collections import OrderedDict
import numpy as np
import theano
from theano import tensor as T
import warnings

from . import distributions
from . import init_rngs, init_weights, Layer, norm_weight, resolve_class
from ..utils import (
    concatenate,
    floatX
)


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

    _components = ['distribution']
    _arg_map = {
        'input': 'dim_in',
        'X': 'dim_in',
        'output': 'dim_out',
        'P': 'dim_out',
        'Z': 'dim_out',
        'H_0': 'dim_hs[0]',
        'H_N': 'dim_hs[-1]',
        'G_0': 'dim_hs[0]',
        'G_N': 'dim_hs[-1]'
    }
    _help = {
        'X': 'Input tensor',
        'P': 'Output distribution or output',
        'Z': 'Output preactivation',
        'H_{layer}': 'Output for hidden layer',
        'G_{layer}': 'Preactivation for hidden layer'
    }

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

        if isinstance(distribution, distributions.Distribution):
            self.distribution = distribution
        elif distribution is not None:
            DC = resolve_class(distribution)
            self.distribution = DC.factory(
                dim=dim_out, conditional=True, **distribution_args)
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
        super(MLP, self).__init__(name=name, **kwargs)

    @classmethod
    def factory(C, dim_in=None, dim_out=None, **kwargs):
        '''MLP factory.

        Convenience function for building MLPs.

        Note::
            Only `MLP` subclass is supported right now.

        Args:
            layer_type (str): string identifies for MLP subclass.
            dim_in (int): input dimension.
            dim_out (int): output dimension.
            **kwargs: construction keyword arguments.

        Returns:
            MLP

        '''
        if dim_in is None or dim_out is None:
            raise TypeError('Both dim in (%r) and dim_out (%r) must be set'
                            % (dim_in, dim_out))
        return C(dim_in, dim_out, **kwargs)

    def sample(self, P, n_samples=1):
        '''Sample from the conditional distribution.

        Args:
            p (T.tensor): probability.
            n_samples (int): number of samples.

        Returns:
            T.tensor: (samples).
            theano.OrderedUpdates: updates.

        '''
        assert self.distribution is not None
        return self.distribution.sample(n_samples, P=P)

    def neg_log_prob(self, X, P, sum_probs=True):
        '''Negative log probability.

        Args:
            X (T.tensor): sample.
            P (T.tensor): probability.
            sum_probs (bool): whether to sum the last axis in neg log prob.

        Returns:
            T.tensor: negative log probabilities.

        '''
        assert self.distribution is not None
        return self.distribution.neg_log_prob(X, P, sum_probs=sum_probs)

    def entropy(self, P):
        '''Entropy function.

        Args:
            P (T.tensor): probability.

        Returns:
            T.tensor: entropies.

        '''
        assert self.distribution is not None
        return self.distribution.entropy(P)

    def get_center(self, P):
        '''
        Args:
            P (T.tensor): distribution tensor.

        Returns:
            T.tensor: center of distribution.

        '''
        assert self.distribution is not None
        return self.distribution.get_center(P)

    def split_prob(self, P):
        '''
        Args:
            P (T.tensor): distribution tensor.

        Returns:
            T.tuple: split distribution.

        '''
        assert self.distribution is not None
        return self.distribution.split_prob(P)

    def set_params(self):
        self.params = OrderedDict()

        for l in xrange(self.n_layers):
            if l == 0:
                dim_in = self.dim_in
            else:
                dim_in = self.dim_hs[l-1]
            dim_out = self.dim_hs[l]

            W = norm_weight(dim_in, dim_out, scale=self.weight_scale,
                            ortho=False)
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

    def dropout(x, act, rate):
        if act == 'T.tanh':
            x_d = self.trng.binomial(x.shape, p=1-rate, n=1,
                                     dtype=x.dtype)
            x = 2. * (x_d * (x + 1.) / 2) / (1 - rate) - 1
        elif act in ['T.nnet.sigmoid', 'T.nnet.softplus', 'lambda x: x']:
            x_d = self.trng.binomial(x.shape, p=1-rate, n=1, dtype=x.dtype)
            x = x * x_d / (1 - rate)
        else:
            raise NotImplementedError('No dropout for %s yet' % activ)
        return x

    def step_call(self, X, *params):
        '''Step feed forward MLP.

        Args:
            x (T.tensor): input.
            use_noise (bool): use noise in call.
            *params: theano shared variables.

        Returns:
            OrderedDict: results at every layer.

        '''
        params = list(params)
        outs = OrderedDict(X=X)
        outs['input'] = X
        for l in xrange(self.n_layers):
            W = params.pop(0)
            b = params.pop(0)

            if self.weight_noise and self.noise_switch():
                self.logger.debug(
                    'Using weight noise in layer %d for MLP %s' % (l, self.name))
                W_n = W + self.trng.normal(
                    avg=0., std=self.weight_noise, size=W.shape)
                preact = T.dot(X, W_n) + b
            else:
                preact = T.dot(X, W) + b

            if l < self.n_layers - 1:
                X = eval(self.h_act)(preact)
                outs['G_%d' % l] = preact
                outs['H_%d' % l] = X
                if self.dropout and self.noise_switch():
                    self.logger.debug('Adding dropout to layer {layer} for MLP '
                                  '`{name}`'.format(layer=l, name=self.name))
                    X = self.dropout(X, self.h_act, self.dropout)
            else:
                if self.distribution is not None:
                    X = self.distribution(preact)
                else:
                    X = eval(self.h_act)(preact)
                outs['Z'] = preact
                outs['P'] = X
                outs['output'] = X

        assert len(params) == 0, params
        return outs

    def __call__(self, X):
        '''Call function.

        Args:
            X (T.tensor): input.

        Returns:
            OrderedDict: results at every layer.

        '''
        params = self.get_params()
        outs = self.step_call(X, *params)
        return outs

    def get_cost(self, X, P):
        if self.distribution is None:
            cost = ((X - P) ** 2).mean()
        else:
            cost = self.distribution.neg_log_prob(X, P).mean()
        return cost

    def feed(self, X):
        '''Simple feed function.

        Args:
            X (T.tensor): input.

        Returns:
            T.tensor: output

        '''
        return self.__call__(X)['P']

    def step_feed(self, X, *params):
        '''Step feed function.

        Args:
            X (T.tensor): input.
            *params: theano shared variables.

        Returns:
            T.tensor: output

        '''
        return self.step_call(X, *params)['P']

    def preact(self, X):
        '''Simple feed preactivation function.

        Args:
            X (T.tensor): input.

        Returns:
            T.tensor: preactivation

        '''
        return self.__call__(X)['Z']

    def step_preact(self, X, *params):
        '''Step feed preactivation function.

        Args:
            X (T.tensor): input.
            *params: theano shared variables.

        Returns:
            T.tensor: preactivation

        '''
        return self.step_call(X, *params)['Z']


_classes = {'MLP': MLP}