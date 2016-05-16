'''
Module for DARN model.
'''

from collections import OrderedDict
import numpy as np
import theano
from theano import tensor as T

from .distributions import (
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
from . import Layer
from ..utils import floatX
from ..utils.tools import (
    concatenate,
    init_rngs,
    init_weights,
    norm_weight,
    scan
)


class AutoRegressor(Distribution):
    def __init__(self, dim, name='autoregressor', **kwargs):
        self.f_neg_log_prob = _cross_entropy
        self.f_sample = _binomial
        super(AutoRegressor, self).__init__(dim, name=name, must_sample=True,
                                            **kwargs)

    def set_params(self):
        b = np.zeros((self.dim,)).astype(floatX)
        W = norm_weight(self.dim, self.dim, scale=0.001,
                        ortho=False)
        self.params = OrderedDict(W=W, b=b)

    def get_params(self):
        return [self.W, self.b]

    def get_prob(self, x, W, b):
        W = T.tril(W, k=-1)
        p = T.nnet.sigmoid(T.dot(x, W) + b) * 0.9999 + 0.000005
        return p

    def get_L2_weight_cost(self, gamma):
        cost = gamma * (self.W ** 2).sum()
        return cost

    def sample(self, n_samples):
        '''
        Inspired by jbornschein's implementation.
        '''

        z0 = T.zeros((n_samples, self.dim,)).astype(floatX) + T.shape_padleft(self.b)
        rs = self.trng.uniform((self.dim, n_samples), dtype=floatX)

        def _step_sample(i, W_i, r_i, z):
            p_i = T.nnet.sigmoid(z[:, i]) * 0.9999 + 0.000005
            x_i = (r_i <= p_i).astype(floatX)
            z   = z + T.outer(x_i, W_i)
            return z, x_i

        seqs = [T.arange(self.dim), self.W, rs]
        outputs_info = [z0, None]
        non_seqs = []

        (zs, x), updates = scan(_step_sample, seqs, outputs_info, non_seqs,
                                self.dim)

        return x.T, updates

    def step_neg_log_prob(self, x, *params):
        p = self.get_prob(x, *params)
        nlp = -x * T.log(p) - (1 - x) * T.log(1 - p)
        return nlp.sum(axis=nlp.ndim-1)

    def neg_log_prob(self, x):
        return self.step_neg_log_prob(x, *self.get_params())

    def entropy(self):
        return T.constant(0.).astype(floatX)

    def prototype_samples(self, size):
        return self.trng.uniform(size, dtype=floatX)


class DARN(Layer):
    must_sample = True

    def __init__(self, dim_in, dim_h, dim_out, n_layers,
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
        else:
            raise ValueError()

        kwargs = init_weights(self, **kwargs)
        kwargs = init_rngs(self, **kwargs)

        super(DARN, self).__init__(name=name)

    def sample(self, c, n_samples=1, return_probs=False):
        if c.ndim == 1:
            c = c[None, :]
        elif c.ndim > 2:
            raise ValueError()

        x = T.zeros((n_samples, self.dim_out)).astype(floatX)
        z = T.zeros((n_samples, self.dim_out,)).astype(floatX) + self.bar[None, :]
        z = z[None, :, :] + c[:, None, :]
        z = z.reshape((z.shape[0] * z.shape[1], z.shape[2]))
        rs = self.trng.uniform((self.dim_out, z.shape[0]), dtype=floatX)

        def _step_sample(i, W_i, r_i, z):
            p_i = T.nnet.sigmoid(z[:, i])
            x_i = (r_i <= p_i).astype(floatX)
            z += T.outer(x_i, W_i)
            return z, x_i, p_i

        seqs = [T.arange(self.dim_out), self.War, rs]
        outputs_info = [z, None, None]
        non_seqs = []

        (zs, x, p), updates = scan(_step_sample, seqs, outputs_info, non_seqs,
                                self.dim_out, name='darn_sample')

        if c.ndim == 1:
            x = x.T[None, :, :]
            p = p.T[None, :, :]
        else:
            x = x.T
            x = x.reshape((n_samples, x.shape[0] // n_samples, x.shape[1]))
            p = p.T
            p = p.reshape((n_samples, p.shape[0] // n_samples, p.shape[1]))

        if return_probs:
            return p, updates
        else:
            return x, updates

    def step_neg_log_prob(self, x, c, War, bar):
        W = T.tril(War, k=-1)
        p = T.nnet.sigmoid(T.dot(x, W) + bar + c)
        return self.f_neg_log_prob(x, p)

    def neg_log_prob(self, x, c):
        W = T.tril(self.War, k=-1)
        p = T.nnet.sigmoid(T.dot(x, W) + self.bar + c)
        return self.f_neg_log_prob(x, p)

    def entropy(self, p):
        return self.f_entropy(p)

    def prob(self, p):
        return p

    @staticmethod
    def factory(dim_in=None, dim_h=None, dim_out=None, n_layers=None,
                **kwargs):
        return DARN(dim_in, dim_h, dim_out, n_layers, **kwargs)

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

        dim_in = self.dim_in
        dim_out = self.dim_h

        for l in xrange(self.n_layers):
            if l > 0:
                dim_in = self.dim_h
            if l == self.n_layers - 1:
                dim_out = self.dim_out

            W = norm_weight(dim_in, dim_out,
                            scale=self.weight_scale, ortho=False)
            b = np.zeros((dim_out,)).astype(floatX)

            self.params['W%d' % l] = W
            self.params['b%d' % l] = b

        b = np.zeros((self.dim_out,)).astype(floatX)
        W = norm_weight(self.dim_out, self.dim_out, scale=self.weight_scale,
                                ortho=False)

        self.params['War'] = W
        self.params['bar'] = b

    def get_params(self):
        params = []
        for l in xrange(self.n_layers):
            W = self.__dict__['W%d' % l]
            b = self.__dict__['b%d' % l]
            params += [W, b]

        params += [self.War, self.bar]
        return params

    def preact(self, x, *params):
        raise NotImplementedError()

    def step_call(self, x, *params):
        # Used within scan with `get_params`
        params = list(params)

        for l in xrange(self.n_layers):
            W = params.pop(0)
            b = params.pop(0)

            if l == self.n_layers - 1:
                x = T.dot(x, W) + b
            else:
                activ = self.h_act
                x = eval(activ)(T.dot(x, W) + b)

        assert len(params) == 2, params
        return x

    def __call__(self, x, return_preact=False):
        params = self.get_params()
        x = self.step_call(x, *params)

        return x
