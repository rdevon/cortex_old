"""
Module for GRU layers
"""

import copy
from collections import OrderedDict
from layers import FFN
from layers import Layer
import numpy as np
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import tools
from tools import gaussian
from tools import log_gaussian


floatX = 'float32'#theano.config.floatX

def raise_type_error(o, t):
    raise ValueError('%s is not of type %s' % (type(o), t))

pi = theano.shared(np.pi).astype(floatX)


class RNN(Layer):

    def __init__(self, dim_in, dim_h, name='rnn'):
        self.dim_in = dim_in
        self.dim_h = dim_h
        super(RNN, self).__init__(name)

    def recurrent_step(self, *xs):
        preact = T.sum(xs, axis=0)
        return preact

    @staticmethod
    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

class StackedRNN(RNN):
    def __init__(self, name='stacked_rnn', forward_backward=True, *rnns):
        self.forward_backward = forward_backward
        self.rnns = OrderedDict()
        for rnn in rnns:
            k = rnn.name
            if not isinstance(rnn, RNN):
                raise_type_error(rnn, RNN)
            if k in self.rnns.keys():
                raise ValueError('Duplicate RNN name %s' % k)
            self.rnns[k] = rnn
        super(StackedRNN, self).__init__()

    def __call__(self, state_below):
        direction = False
        rval = OrderedDict()

        stack_rvals = OrderedDict()
        for rnn_k, rnn in self.rnns.iteritems():
            if direction:
                x = state_below[::-1]
            else:
                x = state_below

            r = rnn(x)
            for k, v in r.iteritems():
                if k in stack_rvals.keys():
                    stack_rvals[k][rnn_k] = v
                else:
                    stack_rvals[k] = OrderedDict(rnn_k=v)
                rval[k + '_' + rnn_k] = v

            if self.forward_backward:
                direction = not direction

        for k, v in stack_rvals.iteritems():
            rnns = v.values()
            if len(rnns) == 1:
                rval[k] = rnns[0]
            else:
                rval[k] = tools.concatenate(rnns, axis=2)

        return rval


class GRU(RNN):
    def __init__(self, dim_in, dim_h, weight_noise=False, name='gru'):
        super(GRU, self).__init__(dim_in, dim_h, name)
        self.weight_noise = weight_noise
        self.set_params()

    def set_params(self):
        norm_weight = tools.norm_weight
        ortho_weight = tools.ortho_weight
        W = np.concatenate([norm_weight(self.dim_in, self.dim_h),
                            norm_weight(self.dim_in, self.dim_h)], axis=1)
        b = np.zeros((2 * self.dim_h,)).astype(floatX)
        U = np.concatenate([ortho_weight(self.dim_h),
                            ortho_weight(self.dim_h)], axis=1)
        Wx = norm_weight(self.dim_in, self.dim_h)
        Ux = ortho_weight(self.dim_h)
        bx = np.zeros((self.dim_h,)).astype(floatX)
        self.params = OrderedDict(W=W, b=b, U=U, Wx=Wx, Ux=Ux, bx=bx)

        if self.weight_noise:
            W_noise = (W * 0).astype(floatX)
            U_noise = (U * 0).astype(floatX)
            Wx_noise = (Wx * 0).astype(floatX)
            Ux_noise = (Ux * 0).astype(floatX)
            self.params.update(W_noise=W_noise, U_noise=U_noise,
                               Wx_noise=Wx_noise, Ux_noise=Ux_noise)

    def set_inputs(self, state_below):
        W = self.W + self.W_noise if self.weight_noise else self.W
        Wx = self.Wx + self.Wx_noise if self.weight_noise else self.Wx
        x = T.dot(state_below, W) + self.b
        x_ = T.dot(state_below, Wx) + self.bx
        return x, x_

    def get_gates(self, preact):
        r = T.nnet.sigmoid(RNN._slice(preact, 0, self.dim_h))
        u = T.nnet.sigmoid(RNN._slice(preact, 1, self.dim_h))
        return r, u

    def step_slice(self, x_, xx_, h_, U, Ux, b, bx):
        preact = self.recurrent_step(T.dot(h_, U), x_, b)
        r, u = self.get_gates(preact)
        preactx = T.dot(h_, Ux) * r + xx_ + bx
        h = T.tanh(preactx)
        h = u * h_ + (1. - u) * h
        return h

    def __call__(self, state_below):
        n_steps = state_below.shape[0]
        n_samples = state_below.shape[1]

        x, x_ = self.set_inputs(state_below)

        seqs = [x, x_]
        outputs_info = [T.alloc(0., n_samples, self.dim_h)]
        non_seqs = [U, Ux, b, bx]

        rval, updates = theano.scan(self.step_slice,
                                    sequences=seqs,
                                    outputs_info=outputs_info,
                                    non_sequences=non_seqs,
                                    name=tools._p(self.name, '_layers'),
                                    n_steps=n_steps,
                                    profile=tools.profile,
                                    strict=True)

        return OrderedDict(h=rval), updates


class GenerativeGRU(GRU):
    def __init__(self, dim_in, dim_h, weight_noise=False, name='gen_gru'):
        super(GenerativeGRU, self).__init__(dim_in, dim_h, name=name)
        if weight_noise:
            raise NotImplementedError()
        self.weight_noise = weight_noise
        self.set_params()

    def set_params(self):
        norm_weight = tools.norm_weight
        ortho_weight = tools.ortho_weight
        XHa = np.concatenate([norm_weight(self.dim_in, self.dim_h),
                             norm_weight(self.dim_in, self.dim_h)], axis=1)
        bha = np.zeros((2 * self.dim_h,)).astype(floatX)
        Ura = np.concatenate([ortho_weight(self.dim_h),
                              ortho_weight(self.dim_h)], axis=1)

        XHb = norm_weight(self.dim_in, self.dim_h)
        bhb = np.zeros((self.dim_h,)).astype(floatX)
        Urb = ortho_weight(self.dim_h)

        HX = norm_weight(self.dim_h, self.dim_in)
        bx = np.zeros((self.dim_in,)).astype(floatX)

        self.params = OrderedDict(XHa=XHa, bha=bha, Ura=Ura, XHb=XHb, bhb=bhb,
                                  Urb=Urb, HX=HX, bx=bx)

        if self.weight_noise:
            XHa_noise = (XHa * 0).astype(floatX)
            Ura_noise = (Ura * 0).astype(floatX)
            XHb_noise = (XHb * 0).astype(floatX)
            Urb_noise = (Urb * 0).astype(floatX)
            HX_noise = (HX * 0).astype(floatX)
            self.params.update(XHa_noise=XHa_noise, Ura_noise=Ura_noise,
                               XHb_noise=XHb_noise, Urb_noise=Urb_noise,
                               HX_noise=HX_noise)

    def step_slice(self, x_, p_, h_, XHa, Ura, bha, XHb, Urb, bhb, HX, bx):
        preact = T.dot(h_, Ura) + T.dot(x_, XHa) + bha
        r, u = self.get_gates(preact)
        preactx = T.dot(h_, Urb) * r + T.dot(x_, XHb, bha) + bhb
        h = T.tanh(preactx)
        h = u * h_ + (1. - u) * h
        p = T.nnet.sigmoid(T.dot(h, HX) + bx)
        x = trng.binomial(p=p, size=p.shape, n=1, dtype=p.dtype)
        return x, p, h

    def get_non_seqs(self):
        return [self.XHa, self.Ura, self.bha, self.XHb, self.Urb, self.bhb,
                self.HX, self.bx]

    def __call__(self, x0=None, n_samples=10, n_steps=10):
        dim_in, dim_h = self.XH.shape
        n_steps = state_below.shape[0]
        n_samples = state_below.shape[1]

        if x0 is None:
            x0 = trng.binomial(size=(n_samples, dim_in), p=0.5, n=1, dtype='float32')
            p0 = T.unbroadcast(T.alloc(0.5, n_samples, dim_in), 0)
            h0 = T.nnet.tanh(T.dot(x0, XHa) + bha)
        else:
            p0 = T.zeros_like(x0) + x0
            h0 = T.alloc(0., n_samples, self.dim_h)

        seqs = []
        outputs_info = [x0, p0, h0]
        non_seqs = self.get_non_seqs()

        (x, p, h), updates = theano.scan(self.step_slice,
                                    sequences=seqs,
                                    outputs_info=outputs_info,
                                    non_sequences=non_seqs,
                                    name=tools._p(self.name, '_layers'),
                                    n_steps=n_steps,
                                    profile=tools.profile,
                                    strict=True)
        x = tools.concatenate([x0[None, :], x])
        p = tools.concatenate([p0[None, :], p])
        h = tools.concatenate([h0[None, :], h])

        return OrderedDict(x=x, p=p, h=h, x0=x0, p0=p0, h0=h0), updates


class GenStochasticGRU(GenerativeGRU):
    def __init__(self, dim_in, dim_h, weight_noise=False, name='gen_stoch_gru',
                 trng=None, stochastic=True):
        self.stochastic = stochastic
        if trng is None:
            self.trng = RandomStreams(6 * 10 * 2015)
        else:
            self.trng = trng
        super(GenStochasticGRU, self).__init__(dim_in, dim_h,
                                               weight_noise=weight_noise,
                                               name=name)

    def set_params(self):
        super(GenStochasticGRU, self).set_params()
        sigmas = np.ones((self.dim_h,)).astype(floatX)
        self.params.update(sigmas=sigmas)

    def sample_x(self, x, z, zp, mask, HX, bx, sigmas, *params):
        p = T.nnet.sigmoid(T.dot(z, HX) + bx)
        '''
        y = T.switch(mask, 1., x).astype(floatX)
        h_ = T.zeros_like(z).astype(floatX)
        _, _, h, _ = self.step_slice(y, p, h_, z, *params)
        q1 = gaussian(zp, h, sigmas)

        y = T.switch(mask, 0., x).astype(floatX)
        _, _, h, _ = self.step_slice(y, p, h_, z, *params)
        q2 = gaussian(zp, h, sigmas)

        prob =
        return T.switch(mask, self.trng.binomial(p=prob, size=x.shape, n=1,
                                                 dtype=x.dtype), x), prob, x, p, q1, q2
        '''
        sample = self.trng.binomial(p=p, size=p.shape, n=1, dtype=p.dtype)
        x = T.set_subtensor(x[1:-1], sample[1:-1])
        return x, p

    def sample_z(self, z, z_, zp, x_, x, mask, l, HX, bx, sigmas, *params):
        h_ = T.zeros_like(z_)
        p_ = T.zeros_like(x_)

        _, _, h, _ = self.step_slice(x_, p_, h_, z_, *params)
        log_prob_1 = log_gaussian(z, h, sigmas).mean()

        _, _, hp, _ = self.step_slice(x, p_, h, z, *params)
        log_prob_2 = log_gaussian(zp, hp, sigmas).mean()

        y = T.dot(z, HX) + bx
        log_prob_3 = (- x * T.nnet.softplus(y) - (1 - x) * T.nnet.softplus(-y)).mean()

        log_prob = log_prob_1 + log_prob_2 + log_prob_3
        grads = theano.grad(log_prob, wrt=z,
                            consider_constant=[z_, zp, x_, x, h_, p_])

        return (z + l * grads).astype(floatX), h, log_prob, z_, zp, x_

    def sample_zx(self, z, z_, zp, x_, x, x_mask, z_mask, l, HX, bx, sigmas, *params):
        z, h, lp, z_, zp, x_ = self.sample_z(z, z_, zp, x_, x, z_mask, l, HX, bx, sigmas, *params)
        x, p = self.sample_x(x, z, zp, x_mask, HX, bx, sigmas, *params)
        return x, z, p, h, lp, z_, zp, x_

    def step_infer(self, x, z, l, HX, bx, sigmas, *params):
        def shift_left(y):
            y_s = T.zeros_like(y)
            y_s = T.set_subtensor(y_s[:-1], y[1:])
            return y

        def shift_right(y):
            y_s = T.zeros_like(y)
            y_s = T.set_subtensor(y_s[1:], y[:-1])
            return y

        z_ = shift_right(z)
        zp = shift_left(z)
        x_ = shift_right(x)
        xp = shift_left(x)

        x_mask = self.trng.binomial(p=0.1, size=x.shape, n=1, dtype='int64')
        x_mask = T.set_subtensor(x_mask[0], 0)
        x_mask = T.set_subtensor(x_mask[-1], 0)
        z_mask = self.trng.binomial(p=0.1, size=z.shape, n=1, dtype='int64')

        return self.sample_zx(z, z_, zp, x_, x, x_mask, z_mask, l, HX, bx,
                              sigmas, *params)

    def inference(self, x, z, l, n_steps=10):
        seqs = []
        non_seqs = [x, z, l, self.HX, self.bx,
                    self.sigmas] + self.get_non_seqs()

        outputs_info = [None, None]
        (xs, zs, hs, ps), updates = theano.scan(
            self.step_infer,
            sequences=seqs,
            outputs_info=outputs_info,
            non_sequences=non_seqs,
            name=tools._p(self.name + '_infer', '_layers'),
            n_steps=n_steps,
            profile=tools.profile,
            strict=True
        )

        updates += [(self.sigmas, (hs - hs.mean(axis=(0, 1))**2).mean(axis=(0, 1)))]

        return (xs[-1], zs[-1], hs[-1], ps[-1]), updates

    def get_energy(self, x, z):
        pass


    def step_slice(self, x_, p_, h_, z_, XHa, Ura, bha, XHb, Urb, bhb, HX, bx, sigmas):
        preact = T.dot(z_, Ura) + T.dot(x_, XHa) + bha
        r, u = self.get_gates(preact)
        preactx = T.dot(z_, Urb) * r + T.dot(x_, XHb) + bhb
        h = T.tanh(preactx)
        h = u * z_ + (1. - u) * h
        z = self.trng.normal(size=h.shape, avg=h, std=sigmas, dtype=h.dtype)
        p = T.nnet.sigmoid(T.dot(z, HX) + bx)
        x = self.trng.binomial(p=p, size=p.shape, n=1, dtype=p.dtype)
        return x, p, h, z

    def get_non_seqs(self):
        return [self.XHa, self.Ura, self.bha, self.XHb, self.Urb, self.bhb,
                self.HX, self.bx, self.sigmas]

    def __call__(self, x0=None, xT=None, n_steps=10):
        dim_in, dim_h = self.XHa.shape
        n_samples = x0.shape[0]

        if x0 is None:
            x0 = trng.binomial(size=(n_samples, dim_in), p=0.5, n=1,
                               dtype='float32')
            p0 = T.unbroadcast(T.alloc(0.5, n_samples, dim_in), 0)
            h0 = T.nnet.tanh(T.dot(x0, self.XHa) + bha)
        else:
            p0 = T.zeros_like(x0) + x0
            h0 = T.alloc(0., n_samples, self.dim_h)
        z0 = self.trng.normal(size=h0.shape, avg=h0, std=self.sigmas[None, :],
                              dtype='float32')

        seqs = []
        outputs_info = [x0, p0, h0, z0]
        non_seqs = self.get_non_seqs()

        (x, p, h, z), updates = theano.scan(self.step_slice,
            sequences=seqs,
            outputs_info=outputs_info,
            non_sequences=non_seqs,
            name=tools._p(self.name, '_layers'),
            n_steps=n_steps,
            profile=tools.profile,
            strict=True)

        x = tools.concatenate([x0[:, None, :].dimshuffle(1, 0, 2), x])
        p = tools.concatenate([p0[:, None, :].dimshuffle(1, 0, 2), p])
        h = tools.concatenate([h0[:, None, :].dimshuffle(1, 0, 2), h])
        z = tools.concatenate([z0[:, None, :].dimshuffle(1, 0, 2), z])

        if xT is not None:
            x = T.set_subtensor(x[-1], xT)

        return OrderedDict(x=x, p=p, h=h, z=z, x0=x0, p0=p0, h0=h0, z0=z0), updates


class CondGenGRU(GenerativeGRU):
    def __init__(self, dim_in, dim_h, weight_noise=False, name='cond_gen_gru',
                 trng=None, stochastic=True):
        self.stochastic = stochastic
        if trng is None:
            self.trng = RandomStreams(6 * 10 * 2015)
        else:
            self.trng = trng
        super(CondGenGRU, self).__init__(dim_in, dim_h, weight_noise=weight_noise,
                                         name=name)

    def set_params(self):
        norm_weight = tools.norm_weight
        super(CondGenGRU, self).set_params()
        Wx0a = np.concatenate([norm_weight(self.dim_in, self.dim_h),
                               norm_weight(self.dim_in, self.dim_h)], axis=1)
        Wx0b = norm_weight(self.dim_in, self.dim_h)
        self.params.update(Wx0a=Wx0a, Wx0b=Wx0b)

    def step_slice(self, x_, p_, h_, x0, XHa, Ura, bha, XHb, Urb, bhb, HX, bx,
                   Wx0a, Wx0b):

        preact = T.dot(h_, Ura) + T.dot(x_, XHa) + T.dot(x0, Wx0a) + bha
        r, u = self.get_gates(preact)

        preactx = T.dot(h_, Urb) * r + T.dot(x_, XHb) + T.dot(x0, Wx0b) + bhb
        h = T.tanh(preactx)
        h = u * h_ + (1. - u) * h
        p = T.nnet.sigmoid(T.dot(h, HX) + bx)
        if self.stochastic:
            x = self.trng.binomial(p=p, size=p.shape, n=1, dtype=p.dtype)
        else:
            x = p
        return x, p, h

    def __call__(self, x0, xT, reversed=True, n_steps=10):
        if reversed:
            x_temp = x0
            x0 = xT
            xT = x_temp

        dim_in, dim_h = self.XHa.shape
        n_samples = x0.shape[0]

        if x0 is None:
            x0 = trng.binomial(size=(n_samples, dim_in), p=0.5, n=1, dtype='float32')
            p0 = T.unbroadcast(T.alloc(0.5, n_samples, dim_in), 0)
            h0 = T.nnet.tanh(T.dot(x0, XHa) + bha)
        else:
            p0 = T.zeros_like(x0) + x0
            h0 = T.alloc(0., n_samples, self.dim_h)

        seqs = []
        outputs_info = [x0, p0, h0]
        non_seqs = [xT, self.XHa, self.Ura, self.bha, self.XHb, self.Urb,
                    self.bhb, self.HX, self.bx, self.Wx0a, self.Wx0b]

        (x, p, h), updates = theano.scan(self.step_slice,
                                    sequences=seqs,
                                    outputs_info=outputs_info,
                                    non_sequences=non_seqs,
                                    name=tools._p(self.name, '_layers'),
                                    n_steps=n_steps,
                                    profile=tools.profile,
                                    strict=True)

        x = tools.concatenate([x0[:, None, :].dimshuffle(1, 0, 2), x])
        p = tools.concatenate([p0[:, None, :].dimshuffle(1, 0, 2), p])
        h = tools.concatenate([h0[:, None, :].dimshuffle(1, 0, 2), h])
        x = T.set_subtensor(x[-1], xT)

        if reversed:
            x = x[::-1]
            p = p[::-1]
            h = h[::-1]

        return OrderedDict(x=x, p=p, h=h, x0=x0, p0=p0, h0=h0, xT=xT), updates
