"""
Module for RNN layers
"""

import copy
from collections import OrderedDict
from gru import RNN
from layers import FFN
from layers import Layer
import numpy as np
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import tools
from tools import gaussian
from tools import log_gaussian


norm_weight = tools.norm_weight
ortho_weight = tools.ortho_weight
floatX = 'float32'#theano.config.floatX

def raise_type_error(o, t):
    raise ValueError('%s is not of type %s' % (type(o), t))

pi = theano.shared(np.pi).astype(floatX)


class GenerativeRNN(RNN):
    def __init__(self, dim_in, dim_h, weight_noise=False, name='gen_rnn',
                 weight_scale=0.1, rng=None, trng=None):
        if weight_noise:
            raise NotImplementedError()

        if trng is None:
            self.trng = RandomStreams(6 * 10 * 2015)
        else:
            self.trng = trng

        self.weight_noise = weight_noise
        self.weight_scale = weight_scale

        if rng is None:
            rng = tools.rng_
        self.rng = rng

        super(GenerativeRNN, self).__init__(dim_in, dim_h, name=name)
        self.set_params()

    def set_params(self):
        XH = norm_weight(self.dim_in, self.dim_h)
        bh = np.zeros((self.dim_h,)).astype(floatX)
        Ur = ortho_weight(self.dim_h)

        HX = norm_weight(self.dim_h, self.dim_in)
        bx = np.zeros((self.dim_in,)).astype(floatX)

        self.params = OrderedDict(XH=XH, bh=bh, Ur=Ur, HX=HX, bx=bx)

        if self.weight_noise:
            XH_noise = (XH * 0).astype(floatX)
            Ur_noise = (Ur * 0).astype(floatX)
            HX_noise = (HX * 0).astype(floatX)
            self.params.update(XH_noise=XH_noise, Ur_noise=Ur_noise,
                               HX_noise=HX_noise)

    def step_slice(self, x_, p_, h_, XH, Ur, bh, HX, bx):
        h = T.tanh(T.dot(x_, XH) + T.dot(h_, Ur) + bh)
        p = T.nnet.sigmoid(T.dot(h, HX) + bx)
        x = self.trng.binomial(p=p, size=p.shape, n=1, dtype=p.dtype)
        return x, p, h

    def get_non_seqs(self):
        return [self.XH, self.Ur, self.bh, self.HX, self.bx]

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()

class GenRNN(GenerativeRNN):
    def __init__(self, dim_in, dim_h, weight_noise=False, name='gen_rnn',
                 h0_mode='average', rate=0.1, trng=None):
        if weight_noise:
            raise NotImplementedError()

        self.h0_mode = h0_mode
        self.rate = rate

        self.weight_noise = weight_noise
        super(GenRNN, self).__init__(dim_in, dim_h, weight_noise=weight_noise,
                                     trng=trng, name=name)

    def set_params(self):
        super(GenRNN, self).set_params()

        if self.h0_mode == 'average':
            h0 = np.zeros((self.dim_h,)).astype(floatX)
            self.params.update(h0=h0)
        elif self.h0_mode == 'ffn':
            W0 = norm_weight(self.dim_in, self.dim_h, scale=self.weight_scale,
                             rng=self.rng)
            b0 = np.zeros((self.dim_h,)).astype('float32')
            self.params.update(W0=W0, b0=b0)
        elif self.h0_mode is not None:
            raise ValueError(self.h0_mode)

    def step_sample(self, x_, p_, h_, XH, Ur, bh, HX, bx):
        h = T.tanh(T.dot(x_, XH) + T.dot(h_, Ur) + bh)
        p = T.nnet.sigmoid(T.dot(h, HX) + bx)
        x = self.trng.binomial(p=p, size=p.shape, n=1, dtype=p.dtype)

        return x, p, h

    def step_slice(self, m_, x_, h_, Ur, HX, bx):
        h = T.tanh(T.dot(h_, Ur) + x_)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_
        p = T.nnet.sigmoid(T.dot(h, HX) + bx)
        x = self.trng.binomial(p=p, size=p.shape, n=1, dtype=p.dtype)

        return h, x, p

    def sample(self, x0=None, n_samples=10, n_steps=10):
        dim_in, dim_h = self.XH.shape

        if x0 is None:
            x0 = self.trng.binomial(size=(n_samples, dim_in), p=0.5, n=1, dtype='float32')
            p0 = T.unbroadcast(T.alloc(0.5, n_samples, dim_in), 0)
        else:
            p0 = T.zeros_like(x0) + x0

        if self.h0_mode == 'average':
            h0 = T.alloc(0., x0.shape[0], self.dim_h).astype(floatX) + self.h0[None, :]
        elif self.h0_mode == 'ffn':
            h0 = T.dot(x0, self.W0) + self.b0
        elif self.h0_mode is None:
            h0 = T.alloc(0., n_samples, self.dim_h)
        else:
            raise ValueError(self.h0_mode)

        seqs = []
        outputs_info = [x0, p0, h0]
        non_seqs = self.get_non_seqs()

        (x, p, h), updates = theano.scan(self.step_sample,
                                    sequences=seqs,
                                    outputs_info=outputs_info,
                                    non_sequences=non_seqs,
                                    name=tools._p(self.name, '_sampling'),
                                    n_steps=n_steps,
                                    profile=tools.profile,
                                    strict=True)

        x = tools.concatenate([x0[None, :, :], x])
        p = tools.concatenate([p0[None, :, :], p])
        h = tools.concatenate([h0[None, :, :], h])

        return OrderedDict(x=x, p=p, h=h, x0=x0, p0=p0, h0=h0), updates

    def __call__(self, state_below, mask=None):
        state_below = state_below * (1 - self.trng.binomial(
            p=0.1, size=state_below.shape, n=1, dtype=state_below.dtype))
        n_steps = state_below.shape[0]
        n_samples = state_below.shape[1]

        if mask is None:
            mask = T.alloc(1, n_steps, 1).astype(floatX)

        x = T.dot(state_below, self.XH) + self.bh

        if self.h0_mode == 'average':
            h0 = T.alloc(0., state_below[0].shape[0], self.dim_h).astype(floatX) + self.h0[None, :]
        elif self.h0_mode == 'ffn':
            h0 = T.dot(state_below[0], self.W0) + self.b0
        elif self.h0_mode is None:
            h0 = T.alloc(0., n_samples, self.dim_h)
        else:
            raise ValueError(self.h0_mode)

        seqs = [mask, x]
        outputs_info = [h0, None, None]
        non_seqs = [self.Ur, self.HX, self.bx]

        (h, x, p), updates = theano.scan(self.step_slice,
                                        sequences=seqs,
                                        outputs_info=outputs_info,
                                        non_sequences=non_seqs,
                                        name=tools._p(self.name, '_layers'),
                                        n_steps=n_steps,
                                        profile=tools.profile,
                                        strict=True)

        if self.h0_mode == 'average':
            h0_mean = h.mean(axis=(0, 1))
            new_h = (1. - self.rate) * self.h0 + self.rate * h0_mean
            updates += [(self.h0, new_h)]

        return OrderedDict(h=h, x=x, p=p), updates


class SimpleInferRNN(GenerativeRNN):
    def __init__(self, dim_in, dim_h, name='simple_inference_rnn',
                 h0_mode='average', weight_noise=False, trng=None,
                 stochastic=True, rate=0.1):
        self.stochastic = stochastic
        self.rate = rate
        self.h0_mode = h0_mode

        if trng is None:
            self.trng = RandomStreams(6 * 10 * 2015)
        else:
            self.trng = trng

        super(SimpleInferRNN, self).__init__(dim_in, dim_h,
                                             weight_noise=weight_noise,
                                             name=name)

    def set_params(self):
        super(SimpleInferRNN, self).set_params()

        k = np.int64(0)
        self.params.update(k=k)

        if self.h0_mode == 'average':
            h0 = np.zeros((self.dim_h, )).astype(floatX)
            self.params.update(h0=h0)
        elif self.h0_mode == 'ffn':
            W0 = norm_weight(self.dim_in, self.dim_h, scale=self.weight_scale,
                             rng=self.rng)
            U0 = norm_weight(self.dim_in, self.dim_h, scale=self.weight_scale,
                             rng=self.rng)
            b0 = np.zeros((self.dim_h,)).astype('float32')
            self.params.update(W0=W0, U0=U0, b0=b0)
        else:
            raise ValueError(self.h0_mode)

    def energy(self, x, p):
        return -(x * T.log(p + 1e-7) +
                (1 - x) * T.log(1 - p + 1e-7)).sum(axis=(0, 2))

    def energy_single(self, x, p):
        return -(x * T.log(p + 1e-7) +
                (1 - x) * T.log(1 - p + 1e-7)).sum(axis=0).mean()

    def step_slice(self, m, xx, x_, p_, h_, XH, Ur, bh, HX, bx):
        h = T.tanh(T.dot(h_, Ur) + T.dot(x_, XH) + bh)
        p = T.nnet.sigmoid(T.dot(h, HX) + bx)

        x_p = self.trng.binomial(p=p, size=p.shape, n=1, dtype=p.dtype)
        x = (1. - m) * x_p + m * xx

        return x, p, h

    def step_h(self, x_, h_, XH, Ur, bh, HX, bx):
        h = T.tanh(T.dot(h_, Ur) + T.dot(x_, XH) + bh)
        return h

    def move_h(self, h0, x, l, HX, bx, *params):
        h1 = self.step_h(x[0], h0, *params)
        h = T.concatenate([h0[None, :, :], h1[None, :, :]], axis=0)
        p = T.nnet.sigmoid(T.dot(h, HX) + bx)
        energy = (self.energy(x[0], p[0]) + self.energy(x[2], p[2])).mean()
        grad = theano.grad(energy, wrt=h0, consider_constant=[x])
        h0 = h0 - l * grad
        return h0

    def move_h_single(self, h, x, l, HX, bx):
        p = T.nnet.sigmoid(T.dot(h, HX) + bx)
        energy = self.energy_single(x, p)
        grads = theano.grad(energy, wrt=h, consider_constant=[x])
        return (h - l * grads).astype(floatX), p

    def step_infer(self, h0, m, x, l, HX, bx, *params):
        h0 = self.move_h(h0, x, l, HX, bx, *params)
        h1 = self.step_h(x[0], h0, *params)
        h = T.concatenate([h0[None, :, :], h1[None, :, :]], axis=0)
        p = T.nnet.sigmoid(T.dot(h, HX) + bx)

        x_hat = self.trng.binomial(p=p, size=p.shape, n=1, dtype=p.dtype)
        energy = self.energy(x, p)

        return h0, x_hat, p, energy

    def inference(self, x, mask, l, n_inference_steps=1, max_k=10, mode='noise'):
        # Initializing the variables
        x0 = x[0]
        x1 = x[1]

        updates = theano.OrderedUpdates()
        if mode == 'noise':
            x0 = x0 * (1 - self.trng.binomial(p=0.1, size=x0.shape, n=1,
                                              dtype=x0.dtype))
            x_n = T.concatenate([x0[None, :, :], x1[None, :, :]], axis=0)
        if mode == 'noise_all':
            x_n = x * (1 - self.trng.binomial(p=0.1, size=x.shape, n=1,
                                              dtype=x.dtype))
        elif mode == 'run':
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

        h1 = self.step_h(x0, h0, *self.get_non_seqs())
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
            new_h = (1. - self.rate) * self.h0 + self.rate * h0_mean
            updates += [(self.h0, new_h)]

        return (x_hats, h0s, energy), updates

    def step_sample(self, x_, p_, h_, XH, Ur, bh, HX, bx):
        h = T.tanh(T.dot(h_, Ur) + T.dot(x_, XH) + bh)
        p = T.nnet.sigmoid(T.dot(h, HX) + bx)
        x = self.trng.binomial(p=p, size=p.shape, n=1, dtype=p.dtype)
        return x, p, h

    def sample(self, x0=None, x1=None, l=0.01, n_steps=100, n_samples=1,
               n_inference_steps=100):
        if x0 is None:
            x0 = self.trng.binomial(p=0.5, size=(n_samples, self.dim_in), n=1,
                                    dtype=floatX)

        if self.h0_mode == 'average':
            h0 = T.alloc(0., x0.shape[0],
                         self.dim_h).astype(floatX) + self.h0[None, :]
        elif self.h0_mode == 'ffn':
            if x1 is None:
                x1 = np.zeros(x0) + x0
            h0 = T.dot(x0, self.W0) + T.dot(x1, self.U0) + self.b0

        p0 = T.zeros_like(x0) + x0

        seqs = []
        outputs_info = [x0, p0, h0]
        non_seqs = self.get_non_seqs()

        (xs, ps, hs), updates = theano.scan(
            self.step_sample,
            sequences=seqs,
            outputs_info=outputs_info,
            non_sequences=non_seqs,
            name=tools._p(self.name, 'sample_chains'),
            n_steps=n_steps,
            profile=tools.profile,
            strict=True
        )

        xs = T.concatenate([x0[None, :, :], xs], axis=0)
        ps = T.concatenate([p0[None, :, :], ps], axis=0)

        return (xs, ps), updates


class OneStepInferRNN(SimpleInferRNN):
    def __init__(self, dim_in, dim_h, name='simple_inference_rnn',
                 h0_mode='average', weight_noise=False, trng=None,
                 stochastic=True, rate=0.1, x_init='average'):

        self.x_init = x_init

        super(OneStepInferRNN, self).__init__(dim_in, dim_h, h0_mode=h0_mode,
                                              rate=rate, trng=trng,
                                              stochastic=stochastic,
                                              weight_noise=weight_noise,
                                              name=name)

    def set_params(self):
        super(OneStepInferRNN, self).set_params()

        if self.x_init == 'average':
            xi = np.zeros((self.dim_in, )).astype(floatX)
            self.params.update(xi=xi)
        elif self.x_init == 'ffn':
            Q0 = norm_weight(self.dim_in, self.dim_in, scale=self.weight_scale,
                             rng=self.rng)
            R0 = norm_weight(self.dim_in, self.dim_in, scale=self.weight_scale,
                             rng=self.rng)
            c0 = np.zeros((self.dim_in,)).astype('float32')
            self.params.update(Q0=Q0, R0=R0, c0=c0)

    def move_h(self, h0, x, l, HX, bx, *params):
        h1 = self.step_h(x[0], h0, *params)
        h2 = self.step_h(x[1], h1, *params)
        h = T.concatenate([h0[None, :, :], h1[None, :, :], h2[None, :, :]], axis=0)
        p = T.nnet.sigmoid(T.dot(h, HX) + bx)
        energy = self.energy(x, p).mean()
        grad = theano.grad(energy, wrt=h0, consider_constant=[x])
        h0 = h0 - l * grad
        return h0

    def step_infer(self, h0, x_hat, m, x, l, HX, bx, *params):
        x = T.set_subtensor(x[1], x_hat[1])
        h0 = self.move_h(h0, x, l, HX, bx, *params)
        h1 = self.step_h(x[0], h0, *params)
        h2 = self.step_h(x[1], h1, *params)

        h = T.concatenate([h0[None, :, :], h1[None, :, :], h2[None, :, :]], axis=0)
        p = T.nnet.sigmoid(T.dot(h, HX) + bx)

        x_hat = self.trng.binomial(p=p, size=p.shape, n=1, dtype=p.dtype)
        x_f = T.concatenate([x[0][None, :, :], x[2][None, :, :]], axis=0)
        p_f = T.concatenate([p[0][None, :, :], p[2][None, :, :]], axis=0)
        energy = self.energy(x_f, p_f)

        return h0, x_hat, p, energy

    def inference(self, x, mask, l, n_inference_steps=1, max_k=10, mode='noise'):
        # Initializing the variables
        x0 = x[0]
        x2 = x[1]

        if self.x_init == 'mean':
            x1 = .5 * (x0 + x2)
        elif self.x_init == 'x0':
            x1 = T.zeros_like(x0) + x0
        elif self.x_init == 'x2':
            x1 = T.zeros_like(x2) + x2
        elif self.x_init == 'average':
            x1 = T.alloc(0., x0.shape[0], self.dim_in).astype(floatX) + self.xi[None, :]
        elif self.x_init == 'ffn':
            x1 = T.dot(x0, self.Q0) + T.dot(x2, self.R0) + self.c0
        elif self.x_init == None:
            x1 = self.trng.binomial(p=0.5, size=x0.shape, n=1, dtype=x0.dtype)
        elif self.x_init == 'zero':
            x1 = T.zeros_like(x0)
        elif self.x_init == 'run':
            if self.h0_mode == 'average':
                h0 = T.alloc(0., x0.shape[0], self.dim_h).astype(floatX) + self.h0[None, :]
            elif self.h0_mode == 'ffn':
                h0 = T.dot(x0, self.W0) + T.dot(x2, self.U0) + self.b0
            _, x1, _ = self.step_slice(0., x0[None, :, :], x0, x0, h0, *self.get_non_seqs())
        else:
            raise ValueError()

        x = T.concatenate([x0[None, :, :],
                           x1[None, :, :],
                           x2[None, :, :]], axis=0)

        updates = theano.OrderedUpdates()
        if mode == 'noise':
            x0 = x0 * (1 - self.trng.binomial(p=0.1, size=x0.shape, n=1,
                                              dtype=x0.dtype))
            x_n = T.concatenate([x0[None, :, :],
                                 x1[None, :, :],
                                 x2[None, :, :]], axis=0)
        if mode == 'noise_all':
            x_n = x * (1 - self.trng.binomial(p=0.1, size=x.shape, n=1,
                                              dtype=x.dtype))
        elif mode == 'run':
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
            x_n = T.concatenate([x0[None, :, :], x1[None, :, :], x2[None, :, :]], axis=0)

        if self.h0_mode == 'average':
            h0 = T.alloc(0., x0.shape[0], self.dim_h).astype(floatX) + self.h0[None, :]
        elif self.h0_mode == 'ffn':
            h0 = T.dot(x0, self.W0) + T.dot(x2, self.U0) + self.b0

        h1 = self.step_h(x0, h0, *self.get_non_seqs())
        h2 = self.step_h(x1, h1, *self.get_non_seqs())

        h = T.concatenate([h0[None, :, :], h1[None, :, :], h2[None, :, :]], axis=0)
        p = T.nnet.sigmoid(T.dot(h, self.HX) + self.bx)
        x_hat = self.trng.binomial(p=p, size=p.shape, n=1, dtype=p.dtype)

        seqs = []
        outputs_info = [h0, x_n, None, None]
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

        #energy = self.energy(x, ps[-1])
        x_f = T.concatenate([x[0][None, :, :], x[2][None, :, :]], axis=0)
        p_f = T.concatenate([ps[-1][0][None, :, :], ps[-1][2][None, :, :]], axis=0)
        energy = self.energy(x_f, p_f)

        h0s = T.concatenate([h0[None, :, :], h0s], axis=0)
        x_hats = T.concatenate([x[None, :, :, :],
                                x_hat[None, :, :, :],
                                x_hats], axis=0)
        ps = T.concatenate([x[None, :, :, :],
                            p[None, :, :, :],
                            ps], axis=0)

        if self.h0_mode == 'average':
            h0_mean = h0s[-1].mean(axis=0)
            new_h = (1. - self.rate) * self.h0 + self.rate * h0_mean
            updates += [(self.h0, new_h)]

        if self.x_init == 'average':
            x_mean = x_hats[-1].mean(axis=(0, 1))
            new_x = (1. - self.rate) * self.xi + self.rate * x_mean
            updates += [(self.xi, new_x)]

        return (x_hats, ps, h0s, energy), updates