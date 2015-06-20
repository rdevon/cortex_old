'''
Module for RBM class
'''

from collections import OrderedDict
import numpy as np
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from layers import Layer
import tools


floatX = theano.config.floatX

class RBM(Layer):
    def __init__(self, dim_in, dim_h, name='rbm', trng=None, stochastic=True):
        self.stochastic = stochastic
        self.dim_in = dim_in
        self.dim_h = dim_h
        if trng is None:
            self.trng = RandomStreams(6 * 10 * 2015)
        else:
            self.trng = trng
        super(RBM, self).__init__(name)
        self.set_params()

    def set_params(self):
        norm_weight = tools.norm_weight
        ortho_weight = tools.ortho_weight
        W = norm_weight(self.dim_in, self.dim_h)
        b = np.zeros((self.dim_in,)).astype(floatX)
        c = np.zeros((self.dim_h,)).astype(floatX)

        self.params = OrderedDict(W=W, b=b, c=c)

    def step_energy(self, x_, x, e_, W, b, c):
        q = T.nnet.sigmoid(T.dot(x_, W) + c)
        if self.stochastic:
            z = self.trng.binomial(p=q, size=q.shape, n=1, dtype=q.dtype)
            p = T.nnet.sigmoid(T.dot(z, W.T) + b)
        else:
            p = T.nnet.sigmoid(T.dot(q, W.T) + b)
        return e_ + (x * T.log(p + 1e-7) + (1. - x) * T.log(1. - p + 1e-7)).sum(axis=1)

    def energy(self, x):
        n_steps = x.shape[0]
        x_s = T.zeros_like(x)
        x_s = T.set_subtensor(x_s[:-1], x[1:])

        seqs = [x, x_s]
        outputs_info = [T.alloc(0., x.shape[1]).astype(floatX)]
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

        return OrderedDict(log_p=rval), updates

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
            p0 = T.alloc(.5, n_chains, self.dim_h).astype(floatX)
            x0 = self.trng.binomial(p=self.b,
                                    size=(n_chains, self.dim_in),
                                    n=1, dtype=floatX)
            q0 = T.nnet.sigmoid(T.dot(x0, self.W.T) + self.c)
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