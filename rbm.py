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
    def __init__(self, dim_in, dim_h, name='rbm', rng=None, trng=None,
                 stochastic=True):
        self.stochastic = stochastic
        self.dim_in = dim_in
        self.dim_h = dim_h

        if rng is None:
            rng = tools.rng_
        self.rng = rng

        if trng is None:
            self.trng = RandomStreams(6 * 10 * 2015)
        else:
            self.trng = trng

        super(RBM, self).__init__(name=name)

    def set_params(self):
        W = norm_weight(self.dim_in, self.dim_h, rng=self.rng)
        b = np.zeros((self.dim_in,)).astype(floatX)
        c = np.zeros((self.dim_h,)).astype(floatX)

        self.params = OrderedDict(W=W, b=b, c=c)

    def _step_energy(self, x_, x, e_, W, b, c):
        q = T.nnet.sigmoid(T.dot(x_, W) + c)
        if self.stochastic:
            z = self.trng.binomial(p=q, size=q.shape, n=1, dtype=q.dtype)
            p = T.nnet.sigmoid(T.dot(z, W.T) + b)
        else:
            p = T.nnet.sigmoid(T.dot(q, W.T) + b)
        e = -(x * T.log(p + 1e-7) + (1. - x) * T.log(1. - p + 1e-7))
        e = e.sum(axis=e.ndim-1)
        return e_ + e, e

    def energy(self, x):
        n_steps = x.shape[0] - 1
        x_s = x[1:]
        x = x[:-1]

        seqs = [x, x_s]
        outputs_info = [T.alloc(0., x.shape[1]).astype(floatX), None]
        non_seqs = [self.W, self.b, self.c]

        rval, updates = theano.scan(
            self._step_energy,
            sequences=seqs,
            outputs_info=outputs_info,
            non_sequences=non_seqs,
            name=tools._p(self.name, '_layers'),
            n_steps=n_steps,
            profile=tools.profile,
            strict=True
        )

        return OrderedDict(acc_neg_log_p=rval[0][-1], neg_log_p=rval[1]), updates

    def joint_energy(self, v, h):
        e = -T.dot(T.dot(h, self.W), v.T) - T.dot(v, self.b) - T.dot(h, self.c)
        return e

    def raise_estimate(self, x, n_samples=10, K=100):
        for m in xrange(n_samples):
            q = T.nnet.sigmoid(T.dot(x, self.W) + self.c)
            h_k = self.trng.binomial(p=q, size=q.shape, n=1, dtype=q.dtype)
            energy = self.joint_energy(samples)

            for k in reversed(xrange(K)):
                code


    def _step(self, x_, W, b, c):
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
            h0 = self.trng.binomial(p=q0, size=q0.shape,
                                    n=1, dtype=floatX)
        elif h0 is not None:
            q0 = T.zeros_like(h0) + h0
            p0 = T.nnet.sigmoid(T.dot(h0, self.W.T) + self.b)
            x0 = self.trng.binomial(p=p0,
                                    size=(h0.shape[0], self.dim_in),
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
        outputs_info = [x0, None, None, None]
        non_seqs = [self.W, self.b, self.c]
        (x, h, p, q), updates = theano.scan(
            self._step,
            sequences=seqs,
            outputs_info=outputs_info,
            non_sequences=non_seqs,
            name=tools._p(self.name, '_layers'),
            n_steps=n_steps,
            profile=tools.profile,
            strict=True
        )

        return OrderedDict(x=x, h=h, p=p, q=q), updates


class GradInferRBM(RBM):
    def __init__(self, dim_in, dim_h, name='grad_infer_rbm', h_init_mode=None,
                 trng=None, stochastic=True, param_file=None, learn=True):
        self.h_init_mode = h_init_mode

        super(GradInferRBM, self).__init__(dim_in, dim_h, name=name, learn=learn,
                                           stochastic=stochastic, trng=trng)

    def set_params():
        super(GradInferRBM, self).set_params()

        if self.h_init_mode == 'average':
            h0 = np.zeros((self.dim_h, )).astype(floatX)
            self.params.update(h0=h0)
        elif self.h_init_mode == 'ffn':
            W0 = norm_weight(self.dim_in, self.dim_h, scale=self.weight_scale,
                             rng=self.rng)
            U0 = norm_weight(self.dim_in, self.dim_h, scale=self.weight_scale,
                             rng=self.rng)
            b0 = np.zeros((self.dim_h,)).astype('float32')
            self.params.update(W0=W0, U0=U0, b0=b0)
        elif self.h_init_mode is not None:
            raise ValueError(self.h_init_mode)

    def step_h(self, x, h_, XHa, Ura, bha, XHb, Urb, bhb, HX, bx):
        preact = T.dot(h_, Ura) + T.dot(x, XHa) + bha
        r, u = self.get_gates(preact)
        preactx = T.dot(h_, Urb) * r + T.dot(x, XHb) + bhb
        h = T.tanh(preactx)
        h = u * h_ + (1. - u) * h
        return h

    def move_h(self, h0, x, l, HX, bx, *params):
        h1 = self.step_h(x[0], h0, *params)
        h = T.concatenate([h0[None, :, :], h1[None, :, :]], axis=0)
        p = T.nnet.sigmoid(T.dot(h, HX) + bx)
        energy = self.energy(x, p).mean()
        grad = theano.grad(energy, wrt=h0, consider_constant=[x])
        h0 = h0 - l * grad
        return h0

    def step_infer(self, h0, m, x, l, HX, bx, *params):
        h0 = self.move_h(h0, x, l, HX, bx, *params)
        h1 = self.step_h(x[0], h0, *params)
        h = T.concatenate([h0[None, :, :], h1[None, :, :]], axis=0)
        p = T.nnet.sigmoid(T.dot(h, HX) + bx)
        x_hat = self.trng.binomial(p=p, size=p.shape, n=1, dtype=p.dtype)
        energy = self.energy(x, p)

        return h0, x_hat, p, energy

    def inference(self, x, mask, l, n_inference_steps=1, max_k=10):
        x0 = x[0]
        x1 = x[1]

        if self.h0_mode == 'average':
            h0 = T.alloc(0., x0.shape[0], self.dim_h).astype(floatX) + self.h0[None, :]
        elif self.h0_mode == 'ffn':
            h0 = T.dot(x0, self.W0) + T.dot(x0, self.U0) + self.b0

        p0 = T.nnet.sigmoid(T.dot(h0, self.HX) + self.bx)

        seqs = []
        outputs_info = [x0, p0, h0]
        non_seqs = self.get_non_seqs()

        (xs, ps, hs), updates = theano.scan(
            self.step_sample,
            sequences=seqs,
            outputs_info=outputs_info,
            non_sequences=non_seqs,
            name=tools._p(self.name, 'sample_init'),
            n_steps=max_k,
            profile=tools.profile,
            strict=True
        )

        xs = T.concatenate([x0[None, :, :], xs], axis=0)
        x0 = xs[self.k]
        x_n = T.concatenate([x0[None, :, :], x1[None, :, :]], axis=0)

        if self.h0_mode == 'average':
            h0 = T.alloc(0., x0.shape[0], self.dim_h).astype(floatX) + self.h0[None, :]
        elif self.h0_mode == 'ffn':
            h0 = T.dot(x0, self.W0) + T.dot(x1, self.U0) + self.b0

        h1 = self.step_h(x[0], h0, *self.get_non_seqs())
        h = T.concatenate([h0[None, :, :], h1[None, :, :]], axis=0)
        p = T.nnet.sigmoid(T.dot(h, self.HX) + self.bx)
        x_hat = self.trng.binomial(p=p, size=p.shape, n=1, dtype=p.dtype)

        seqs = []
        outputs_info = [h0, None, None, None]
        non_seqs = [mask, x_n, l, self.HX, self.bx] + self.get_non_seqs()

        (h0s, x_hats, ps, energies), updates_2 = theano.scan(
            self.step_infer,
            sequences=seqs,
            outputs_info=outputs_info,
            non_sequences=non_seqs,
            name=tools._p(self.name, 'infer'),
            n_steps=n_inference_steps,
            profile=tools.profile,
            strict=True
        )
        updates.update(updates_2)

        h0s = T.concatenate([h0[None, :, :], h0s], axis=0)
        x_hats = T.concatenate([x[None, :, :, :],
                                x_hat[None, :, :, :],
                                x_hats], axis=0)
        energy = self.energy(x, ps[-1])

        if self.h0_mode == 'average':
            h0_mean = h0s[-1].mean(axis=0)
            new_h = (1. - self.rate) * h0_mean + self.rate * h0_mean
            updates += [(self.h0, new_h)]

        return (x_hats, h0s, energy), updates