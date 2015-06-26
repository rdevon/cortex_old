'''
Module for RBM class
'''

from collections import OrderedDict
import numpy as np
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import yaml

from layers import Layer
import tools


floatX = theano.config.floatX
norm_weight = tools.norm_weight
ortho_weight = tools.ortho_weight

class RBM(Layer):
    def __init__(self, dim_in, dim_h, name='rbm', trng=None, stochastic=True,
                 param_file=None, learn=True):
        self.stochastic = stochastic
        self.dim_in = dim_in
        self.dim_h = dim_h
        if trng is None:
            self.trng = RandomStreams(6 * 10 * 2015)
        else:
            self.trng = trng
        super(RBM, self).__init__(name=name, learn=learn)
        self.set_params(param_file=param_file)

    def _load(self, param_file):
        param_dict = yaml.load(open(param_file, 'r'))
        params = OrderedDict((k, np.load(v))
            for k, v in param_dict['params'].iteritems())
        return params

    def set_params(self, param_file=None):
        if param_file is None:
            W = norm_weight(self.dim_in, self.dim_h)
            b = np.zeros((self.dim_in,)).astype(floatX)
            c = np.zeros((self.dim_h,)).astype(floatX)
        else:
            params = self._load(param_file)
            W = params['W']
            b = params['b']
            c = params['c']

        self.params = OrderedDict(W=W, b=b, c=c)

    def step_energy(self, x_, x, e_, W, b, c):
        q = T.nnet.sigmoid(T.dot(x_, W) + c)
        if self.stochastic:
            z = self.trng.binomial(p=q, size=q.shape, n=1, dtype=q.dtype)
            p = T.nnet.sigmoid(T.dot(z, W.T) + b)
        else:
            p = T.nnet.sigmoid(T.dot(q, W.T) + b)
        e = (x * T.log(p + 1e-7) + (1. - x) * T.log(1. - p + 1e-7)).sum(axis=1)
        return e_ + e, e

    def energy(self, x):
        n_steps = x.shape[0] - 1
        x_s = x[1:]
        x = x[:-1]

        seqs = [x, x_s]
        outputs_info = [T.alloc(0., x.shape[1]).astype(floatX), None]
        non_seqs = [self.W, self.b, self.c]

        rval, updates = theano.scan(
            self.step_energy,
            sequences=seqs,
            outputs_info=outputs_info,
            non_sequences=non_seqs,
            name=tools._p(self.name, '_layers'),
            n_steps=n_steps,
            profile=tools.profile,
            strict=True
        )

        return OrderedDict(acc_log_p=rval[0], log_p=rval[1]), updates

    def step_slice(self, x_, h_, p_, q_, W, b, c):
        q = T.nnet.sigmoid(T.dot(x_, W) + c)
        h = self.trng.binomial(p=q, size=q.shape, n=1, dtype=q.dtype)
        p = T.nnet.sigmoid(T.dot(h, W.T) + b)
        x = self.trng.binomial(p=p, size=p.shape, n=1, dtype=p.dtype)

        return x, h, p, q

    def __call__(self, n_steps, n_chains=None, x0=None, h0=None):
        assert x0 is None or h0 is None

        if x0 is not None:
            assert n_chains is None
            p0 = T.zeros_like(x0) + x0
            q0 = T.nnet.sigmoid(T.dot(x0, self.W) + self.c)
            h0 = self.tnrg.binomial(p=q0,
                                    size=(n_chains, self.dim_h),
                                    n=1, dtype=floatX)
        elif h0 is not None:
            q0 = T.zeros_like(h0) + h0
            p0 = T.nnet.sigmoid(T.dot(h0, self.W.T) + self.b)
            x0 = self.trng.binomial(p=p0,
                                    size=(n_chains, self.dim_in),
                                    n=1, dtype=floatX)
        else:
            assert n_chains is not None
            p0 = T.alloc(.5, n_chains, self.dim_in).astype(floatX)
            x0 = self.trng.binomial(p=0.5,
                                    size=(n_chains, self.dim_in),
                                    n=1, dtype=floatX)
            q0 = T.nnet.sigmoid(T.dot(x0, self.W) + self.c)
            h0 = self.trng.binomial(p=q0,
                                    size=(n_chains, self.dim_h),
                                    n=1, dtype=floatX)

        seqs = []
        outputs_info = [x0, h0, p0, q0]
        non_seqs = [self.W, self.b, self.c]

        (x, h, p, q), updates = theano.scan(
            self.step_slice,
            sequences=seqs,
            outputs_info=outputs_info,
            non_sequences=non_seqs,
            name=tools._p(self.name, '_layers'),
            n_steps=n_steps,
            profile=tools.profile,
            strict=True
        )

        return OrderedDict(x=x, h=h, p=p, q=q), updates