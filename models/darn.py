'''
Module for DARN model.
'''

from collections import OrderedDict
import numpy as np
from theano import tensor as T

from distributions import (
    Distribution,
    _binomial,
    _cross_entropy,
    _binary_entropy,
    _centered_binomial,
    _normal,
    _neg_normal_log_prob,
    _normal_entropy,
    _normal_prob
)
from layers import Layer
from utils.tools import (
    concatenate,
    floatX,
    init_rngs,
    init_weights,
    norm_weight,
    _scan
)


class AutoRegressor(Distribution):
    def __init__(self, dim, name='autoregressor', **kwargs):
        self.f_neg_log_prob = _cross_entropy
        super(AutoRegressor, self).__init__(dim, name=name, **kwargs)

    def set_params(self):
        b = np.zeros((self.dim,)).astype(floatX)
        W = np.zeros((self.dim, self.dim)).astype(floatX)
        self.params = OrderedDict(W=W, b=b)

    def get_params(self):
        return [self.W, self.b]

    def get_prob(self):
        W = T.tril(self.W, k=-1)
        p = T.nnet.sigmoid(T.dot(x, W) + b)
        return p

    def sample(self, ):
        pass


class DARN(Layer):
    def __init__(self, dim_in, dim_h, dim_out, n_layers,
                 f_sample=None, f_neg_log_prob=None, f_entropy=None,
                 h_act='T.nnet.sigmoid', out_act='T.nnet.sigmoid',
                 name='DARN',
                 **kwargs):

        self.dim_in = dim_in
        self.dim_h = dim_h
        self.dim_out = dim_out
        self.n_layers = n_layers
        assert n_layers > 0

        self.h_act = h_act
        self.out_act = out_act

        if out_act is None:
            out_act = 'T.nnet.sigmoid'

        if out_act == 'T.nnet.sigmoid':
            self.f_sample = _binomial
            self.f_neg_log_prob = _cross_entropy
            self.f_entropy = _binary_entropy
            self.f_prob = lambda x: x
        elif out_act == 'T.tanh':
            self.f_sample = _centered_binomial
        elif out_act == 'lambda x: x':
            self.f_sample = _normal
            self.f_neg_log_prob = _neg_normal_log_prob
            self.f_entropy = _normal_entropy
            self.f_prob = _normal_prob
            self.dim_out *= 2
        else:
            assert f_sample is not None
            assert f_neg_log_prob is not None
            assert out_act is not None

        if f_sample is not None:
            self.sample = f_sample
        if f_neg_log_prob is not None:
            self.net_log_prob = f_neg_log_prob
        if f_entropy is not None:
            self.entropy = f_entropy

        kwargs = init_weights(self, **kwargs)
        kwargs = init_rngs(self, **kwargs)

        #assert len(kwargs) == 0, kwargs.keys()
        super(MLP, self).__init__(name=name)

    def sample(self, p, size=None):
        return self.f_sample(self.trng, p, size=size)

    def neg_log_prob(self, x, p):
        return self.f_neg_log_prob(x, p)

    def entropy(self, p):
        return self.f_entropy(p)

    def prob(self, p):
        return self.f_prob(p)

    @staticmethod
    def factory(dim_in=None, dim_h=None, dim_out=None, n_layers=None,
                **kwargs):
        return MLP(dim_in, dim_h, dim_out, n_layers, **kwargs)

    def get_L2_weight_cost(self, gamma, layers=None):
        if layers is None:
            layers = range(self.n_layers)

        cost = T.constant(0.).astype(floatX)
        for l in layers:
            W = self.__dict__['W%d' % l]
            cost += gamma * (W ** 2).sum()

        return cost

    def set_params(self):
        self.params = OrderedDict()

        for l in xrange(self.n_layers):
            if l == 0:
                dim_in = self.dim_in
            else:
                dim_in = self.dim_h

            if l == self.n_layers - 1:
                dim_out = self.dim_out
            else:
                dim_out = self.dim_h

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

    def preact(self, x, *params):
        # Used within scan with `get_params`
        params = list(params)

        for l in xrange(self.n_layers):
            W = params.pop(0)
            b = params.pop(0)

            if self.weight_noise:
                print 'Using weight noise in layer %d for MLP %s' % (l, self.name)
                W += self.trng.normal(avg=0., std=self.weight_noise, size=W.shape)

            if l == self.n_layers - 1:
                x = T.dot(x, W) + b
            else:
                activ = self.h_act
                x = eval(activ)(T.dot(x, W) + b)

            if self.dropout:
                if activ == 'T.tanh':
                    raise NotImplementedError('dropout for tanh units not implemented yet')
                elif activ in ['T.nnet.sigmoid', 'T.nnet.softplus', 'lambda x: x']:
                    x_d = self.trng.binomial(x.shape, p=1-self.dropout, n=1,
                                             dtype=x.dtype)
                    x = x * x_d / (1 - self.dropout)
                else:
                    raise NotImplementedError('No dropout for %s yet' % activ)

        assert len(params) == 0, params
        return x

    def step_call(self, x, *params):
        x = self.preact(x, *params)
        return eval(self.out_act)(x)

    def __call__(self, x, return_preact=False):
        params = self.get_params()
        if return_preact:
            x = self.preact(x, *params)
        else:
            x = self.step_call(x, *params)

        return x