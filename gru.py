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


norm_weight = tools.norm_weight
ortho_weight = tools.ortho_weight
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

    def set_inputs(self, state_below, suppress_noise=False):
        if self.weight_noise and not suppress_noise:
            W = self.W + self.W_noise
            Wx = self.Wx + self.Wx_noise
        else:
            W = self.W
            Wx = self.Wx
        x = T.dot(state_below, W) + self.b
        x_ = T.dot(state_below, Wx) + self.bx
        return x, x_

    def get_gates(self, preact):
        r = T.nnet.sigmoid(RNN._slice(preact, 0, self.dim_h))
        u = T.nnet.sigmoid(RNN._slice(preact, 1, self.dim_h))
        return r, u

    def get_non_seqs(self):
        return [self.U, self.Ux]

    def step_slice(self, m_, x_, xx_, h_, U, Ux):
        preact = T.dot(h_, U) + x_
        r, u = self.get_gates(preact)
        preactx = T.dot(h_, Ux) * r + xx_
        h = T.tanh(preactx)
        h = u * h_ + (1. - u) * h
        h = m_[:, None] * h + (1. - m_)[:, None] * h_
        return h

    def __call__(self, state_below, mask=None):
        n_steps = state_below.shape[0]
        n_samples = state_below.shape[1]

        if mask is None:
            mask = T.alloc(1., n_steps, 1).astype(floatX)

        x, x_ = self.set_inputs(state_below)

        seqs = [mask, x, x_]
        outputs_info = [T.alloc(0., n_samples, self.dim_h)]
        non_seqs = self.get_non_seqs()

        rval, updates = theano.scan(self.step_slice,
                                    sequences=seqs,
                                    outputs_info=outputs_info,
                                    non_sequences=non_seqs,
                                    name=tools._p(self.name, '_layers'),
                                    n_steps=n_steps,
                                    profile=tools.profile,
                                    strict=True)

        return OrderedDict(h=rval), updates


class GRUWithOutput(GRU):
    def __init__(self, dim_in, dim_h, dim_o, window, convolve_time=False,
                 weight_noise=False, weight_scale=0.01, name='gru_w_output',
                 dropout=False, trng=None, rng=None):
        self.dim_o = dim_o
        self.window = window
        self.convolve_time = convolve_time
        self.dropout = dropout
        if self.dropout and trng is None:
            self.trng = RandomStreams(1234)
        else:
            self.trng = trng

        if rng is None:
            rng = tools.rng_
        self.rng = rng

        super(GRUWithOutput, self).__init__(dim_in, dim_h, name=name,
                                            weight_noise=weight_noise,
                                            weight_scale=weight_scale)

    def set_params(self):
        super(GRUWithOutput, self).set_params()
        Wo = tools.norm_weight(self.dim_h, self.dim_o * self.window,
                               rng=self.rng)
        bo = np.zeros((self.dim_o * self.window,)).astype(floatX)

        self.params.update(OrderedDict(Wo=Wo, bo=bo))

        if self.convolve_time:
            vn = np.ones(self.window).astype(floatX) / 3
            vp = np.ones(self.window).astype(floatX) / 3
            vo = np.ones(self.window).astype(floatX) / 3
            self.params.update(OrderedDict(vn=vn, vp=vp, vo=vo))

        if self.weight_noise:
            Wo_noise = (Wo * 0).astype('float32')
            self.params.update(Wo_noise=Wo_noise)

    def step_slice(self, x_, xx_, h_, U, Ux, Wo, bo):
        preact = self.recurrent_step(T.dot(h_, U), x_)
        r, u = self.get_gates(preact)
        preactx = T.dot(h_, Ux) * r + xx_
        h = T.tanh(preactx)
        h = u * h_ + (1. - u) * h
        o = T.dot(h, Wo) + bo
        return h, o

    def __call__(self, state_below, suppress_noise=False):
        n_steps = state_below.shape[0]
        n_samples = state_below.shape[1]

        x, x_ = self.set_inputs(state_below)

        if self.weight_noise and not suppress_noise:
            U = self.U + self.U_noise
            Ux = self.Ux + self.Ux_noise
            Wo = self.Wo + self.Wo_noise
        else:
            U = self.U
            Ux = self.Ux
            Wo = self.Wo

        seqs = [x, x_]
        outputs_info = [T.alloc(0., n_samples, self.dim_h), None]
        non_seqs = [U, Ux, Wo, self.bo]

        (h, out), updates = theano.scan(self.step_slice,
                                        sequences=seqs,
                                        outputs_info=outputs_info,
                                        non_sequences=non_seqs,
                                        name=tools._p(self.name, '_layers'),
                                        n_steps=n_steps,
                                        profile=tools.profile,
                                        strict=True)

        if self.dropout and not suppress_noise:
            o_d = self.trng.binomial(out.shape, p=1-self.dropout, n=1,
                                     dtype=out.dtype)
            out = out * o_d / (1 - self.dropout)

        if self.convolve_time:
            o_shifted_plus = T.zeros_like(out)
            o_shifted_minus = T.zeros_like(out)
            o_shifted_plus = T.set_subtensor(o_shifted_plus[1:], out[:-1])
            o_shifted_minus = T.set_subtensor(o_shifted_minus[:-1], out[1:])
            out = (o_shifted_minus.dimshuffle(2, 1, 0) * self.vn
                   + out.dimshuffle(2, 1, 0) * self.vo
                   + o_shifted_plus.dimshuffle(2, 1, 0) * self.vp
                   ).dimshuffle(2, 1, 0)

        return OrderedDict(h=h, o=out.sum(axis=0).reshape(
            (out.shape[1], n_steps, self.dim_o)).dimshuffle(1, 0, 2)), updates


class HeirarchalGRU(GRU):
    def __init__(self, dim_in, dim_h, dim_s, weight_noise=False, dropout=False,
                 top_fb=False, bottom_fb=False, trng=None, name='hiero_gru'):
        self.dim_s = dim_s
        self.dropout = dropout
        self.top_fb = top_fb
        self.bottom_fb = bottom_fb
        if self.dropout and trng is None:
            self.trng = RandomStreams(1234)
        else:
            self.trng = trng
        super(HeirarchalGRU, self).__init__(dim_in, dim_h, name=name)

    def set_params(self):
        super(HeirarchalGRU, self).set_params()
        Ws = np.concatenate([norm_weight(self.dim_h, self.dim_s),
                            norm_weight(self.dim_h, self.dim_s)], axis=1)
        bs = np.zeros((2 * self.dim_s,)).astype(floatX)
        Us = np.concatenate([ortho_weight(self.dim_s),
                            ortho_weight(self.dim_s)], axis=1)

        if self.bottom_fb:
            dim_in = 2 * self.dim_h
        else:
            dim_in = self.dim_h
        Wxs = norm_weight(dim_in, self.dim_s)
        Uxs = ortho_weight(self.dim_s)
        bxs = np.zeros((self.dim_s,)).astype(floatX)

        if self.top_fb:
            dim_in = 2 * self.dim_s
        else:
            dim_in = self.dim_s
        Wo = norm_weight(dim_in, 1).astype(floatX)
        bo = np.float32(0.)

        self.params.update(Ws=Ws, bs=bs, Us=Us, Wxs=Wxs, Uxs=Uxs, bxs=bxs,
                           Wo=Wo, bo=bo)

        if self.weight_noise:
            Ws_noise = (Ws * 0).astype(floatX)
            Us_noise = (Us * 0).astype(floatX)
            Wxs_noise = (Wxs * 0).astype(floatX)
            Uxs_noise = (Uxs * 0).astype(floatX)
            self.params.update(Ws_noise=Ws_noise, Us_noise=Us_noise,
                               Wxs_noise=Wxs_noise, Uxs_noise=Uxs_noise)

    def get_gates_lower(self, preact):
        r = T.nnet.sigmoid(RNN._slice(preact, 0, self.dim_h))
        u = T.nnet.sigmoid(RNN._slice(preact, 1, self.dim_h))
        return r, u

    def get_gates_upper(self, preact):
        r = T.nnet.sigmoid(RNN._slice(preact, 0, self.dim_s))
        u = T.nnet.sigmoid(RNN._slice(preact, 1, self.dim_s))
        return r, u

    def step_slice_upper(self, m_, x_, xx_, h_, U, Ux):
        preact = T.dot(h_, U) + x_
        r, u = self.get_gates_upper(preact)
        preactx = T.dot(h_, Ux) * r + xx_
        h = T.tanh(preactx)
        h = u * h_ + (1. - u) * h
        h = m_[:, None] * h + (1. - m_)[:, None] * h_
        return h

    def step_slice_lower(self, m_, x_, xx_, h_, U, Ux):
        # Here we just kill the previous state if the previous token was eof
        # where the mask was 0.
        h_ = m_[:, None] * h_
        preact = T.dot(h_, U) + x_
        r, u = self.get_gates_lower(preact)
        preactx = T.dot(h_, Ux) * r + xx_
        h = T.tanh(preactx)
        h = u * h_ + (1. - u) * h
        return h

    def step_out(self, h, Wo, bo, suppress_noise):
        o = h.dot(Wo) + bo

        if self.dropout and not suppress_noise:
            o_d = self.trng.binomial(o.shape, p=1-self.dropout, n=1,
                                     dtype=o.dtype)
            o = o * o_d / (1 - self.dropout)

        return o

    def __call__(self, state_below, mask=None, suppress_noise=False):
        n_steps = state_below.shape[0]
        n_samples = state_below.shape[1]

        if mask is None:
            mask = T.neq(state_below[:, :, 0], 1).astype(floatX)

        x, x_ = self.set_inputs(state_below, suppress_noise=suppress_noise)

        mask_r = T.roll(mask, 1, axis=0)
        mask_n = T.eq(mask, 0.).astype(floatX)

        seqs = [mask_r, x, x_]
        outputs_info = [T.alloc(0., n_samples, self.dim_h)]
        non_seqs = self.get_non_seqs()

        h, updates = theano.scan(self.step_slice_lower,
                                    sequences=seqs,
                                    outputs_info=outputs_info,
                                    non_sequences=non_seqs,
                                    name=tools._p(self.name, '_layers'),
                                    n_steps=n_steps,
                                    profile=tools.profile,
                                    strict=True)

        if self.bottom_fb:
            mask_r = T.roll(mask[::-1], 1, axis=0)
            seqs = [mask_r, x[::-1], x_[::-1]]
            outputs_info = [T.alloc(0., n_samples, self.dim_h)]
            h_r, updates_r = theano.scan(self.step_slice_lower,
                                    sequences=seqs,
                                    outputs_info=outputs_info,
                                    non_sequences=non_seqs,
                                    name=tools._p(self.name, '_layers_r'),
                                    n_steps=n_steps,
                                    profile=tools.profile,
                                    strict=True)
            h = T.concatenate([h, h_r[::-1]], axis=2)
            updates.update(updates_r)

        if self.weight_noise and not suppress_noise:
            Ws = self.Ws + self.Ws_noise
            Wxs = self.Wxs + self.Wxs_noise
        else:
            Ws = self.Ws
            Wxs = self.Wxs
        x = h.dot(Ws) + self.bs
        x_ = h.dot(Wxs) + self.bxs

        if self.dropout and not suppress_noise:
            x_d = self.trng.binomial(x_.shape, p=1-self.dropout, n=1,
                                     dtype=x_.dtype)
            x = x * T.concatenate([x_d, x_d], axis=2) / (1 - self.dropout)
            x_ = x_ * x_d / (1 - self.dropout)

        seqs = [mask_n, x, x_]
        outputs_info = [T.alloc(0., n_samples, self.dim_s)]
        non_seqs = [self.Us, self.Uxs]

        hs, updates = theano.scan(self.step_slice_upper,
                                    sequences=seqs,
                                    outputs_info=outputs_info,
                                    non_sequences=non_seqs,
                                    name=tools._p(self.name, '_top_layers'),
                                    n_steps=n_steps,
                                    profile=tools.profile,
                                    strict=True)

        if self.top_fb:
            seqs = [mask_n[::-1], x[::-1], x_[::-1]]
            outputs_info = [T.alloc(0., n_samples, self.dim_s)]
            non_seqs = [self.Us, self.Uxs]

            hs_r, updates_r = theano.scan(self.step_slice_upper,
                                        sequences=seqs,
                                        outputs_info=outputs_info,
                                        non_sequences=non_seqs,
                                        name=tools._p(self.name, '_top_layers_r'),
                                        n_steps=n_steps,
                                        profile=tools.profile,
                                        strict=True)
            hs = T.concatenate([hs, hs_r[::-1]], axis=2)
            updates.update(updates_r)

        seqs = [hs]
        outputs_info = [None]
        suppress_noise = 1 if suppress_noise else 0
        non_seqs = [self.Wo, self.bo, suppress_noise]

        o, updates_o = theano.scan(self.step_out,
                                    sequences=seqs,
                                    outputs_info=outputs_info,
                                    non_sequences=non_seqs,
                                    name=tools._p(self.name, '_out'),
                                    profile=tools.profile,
                                    strict=True)
        updates.update(updates_o)

        return OrderedDict(h=h, hs=hs, o=o, mask=mask, mask_n=mask_n), updates


class HeirarchalGRUWO(GRU):
    def __init__(self, dim_in, dim_h, dim_out, weight_noise=False, dropout=False,
                 trng=None, name='heiro_gru_wo'):
        self.dropout = dropout
        self.dim_out = dim_out
        if self.dropout and trng is None:
            self.trng = RandomStreams(1234)
        else:
            self.trng = trng
        super(HeirarchalGRUWO, self).__init__(dim_in, dim_h, name=name)

    def set_params(self):
        super(HeirarchalGRUWO, self).set_params()

        Wo = norm_weight(self.dim_h, self.dim_out).astype(floatX)
        bo = np.float32(0.)

        self.params.update(Wo=Wo, bo=bo)

        if self.weight_noise:
            Wo_noise = (Wo * 0).astype(floatX)
            self.params.update(Wo_noise=Wo_noise)

    def step_slice(self, m_, x_, xx_, h_, U, Ux, Wo, bo):
        # Here we just kill the previous state if the previous token was eof
        # where the mask was 0.
        h_ = m_[:, None] * h_
        preact = T.dot(h_, U) + x_
        r, u = self.get_gates(preact)
        preactx = T.dot(h_, Ux) * r + xx_
        h = T.tanh(preactx)
        h = u * h_ + (1. - u) * h
        o = h.dot(Wo) + bo

        if self.dropout and not suppress_noise:
            o_d = self.trng.binomial(o.shape, p=1-self.dropout, n=1,
                                     dtype=o.dtype)
            o = o * o_d / (1 - self.dropout)

        return h, o

    def __call__(self, state_below, mask=None, suppress_noise=False):
        n_steps = state_below.shape[0]
        n_samples = state_below.shape[1]

        if mask is None:
            mask = T.neq(state_below[:, :, 0], 1).astype(floatX)

        x, x_ = self.set_inputs(state_below)

        mask_r = T.roll(mask, 1, axis=0)
        mask_n = T.eq(mask, 0.).astype(floatX)

        seqs = [mask_r, x, x_]
        outputs_info = [T.alloc(0., n_samples, self.dim_h), None]
        non_seqs = self.get_non_seqs() + [self.Wo, self.bo]

        (h, o), updates = theano.scan(self.step_slice,
                                    sequences=seqs,
                                    outputs_info=outputs_info,
                                    non_sequences=non_seqs,
                                    name=tools._p(self.name, '_layers'),
                                    n_steps=n_steps,
                                    profile=tools.profile,
                                    strict=True)

        n_outs = mask_n[1:].sum().astype('int64')

        o = mask_n[1:].compress(o, axis=0).dimshuffle(1, 0, 2).reshape(
            (o.shape[1], n_outs * self.dim_out))

        return OrderedDict(h=h, o=o, mask=mask, mask_n=mask_n), updates


class GenerativeGRU(GRU):
    def __init__(self, dim_in, dim_h, weight_noise=False, name='gen_gru'):
        super(GenerativeGRU, self).__init__(dim_in, dim_h, name=name)
        if weight_noise:
            raise NotImplementedError()
        self.weight_noise = weight_noise
        self.set_params()

    def set_params(self):
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
        preactx = T.dot(h_, Urb) * r + T.dot(x_, XHb) + bhb
        h = T.tanh(preactx)
        h = u * h_ + (1. - u) * h
        p = T.nnet.sigmoid(T.dot(h, HX) + bx)
        x = self.trng.binomial(p=p, size=p.shape, n=1, dtype=p.dtype)
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


class SimpleInferGRU(GenerativeGRU):
    def __init__(self, dim_in, dim_h, name='simple_inference_gru',
                 weight_noise=False, trng=None, stochastic=True, rate=0.1):
        self.stochastic = stochastic
        self.rate = rate
        if trng is None:
            self.trng = RandomStreams(6 * 10 * 2015)
        else:
            self.trng = trng
        super(SimpleInferGRU, self).__init__(dim_in, dim_h,
                                             weight_noise=weight_noise,
                                             name=name)

    def set_params(self):
        super(SimpleInferGRU, self).set_params()
        h0 = np.zeros((self.dim_h, )).astype(floatX)
        self.params.update(h0=h0)

    def energy(self, x, p):
        return -(x * T.log(p + 1e-7) +
                (1 - x) * T.log(1 - p + 1e-7)).sum(axis=(0, 2))

    def energy_single(self, x, p):
        return -(x * T.log(p + 1e-7) +
                (1 - x) * T.log(1 - p + 1e-7)).sum(axis=0).mean()

    def step_slice(self, m, xx, x_, p_, h_, XHa, Ura, bha, XHb, Urb, bhb, HX, bx):
        preact = T.dot(h_, Ura) + T.dot(x_, XHa) + bha
        r, u = self.get_gates(preact)
        preactx = T.dot(h_, Urb) * r + T.dot(x_, XHb) + bhb
        h = T.tanh(preactx)
        h = u * h_ + (1. - u) * h
        p = T.nnet.sigmoid(T.dot(h, HX) + bx)
        x_p = self.trng.binomial(p=p, size=p.shape, n=1, dtype=p.dtype)
        x = (1. - m) * x_p + m * xx
        return x, p, h

    def step_h(self, x, h_, XHa, Ura, bha, XHb, Urb, bhb, HX, bx):
        preact = T.dot(h_, Ura) + T.dot(x, XHa) + bha
        r, u = self.get_gates(preact)
        preactx = T.dot(h_, Urb) * r + T.dot(x, XHb) + bhb
        h = T.tanh(preactx)
        h = u * h_ + (1. - u) * h
        return h

    #def move_h(self, x_, x, h_, h, l, HX, bx, *params):
        #p_ = T.zeros_like(x_)
        #_, _, h_hat = self.step_slice(1., x, x_, p_, h_, *params)
        #h_hat = T.set_subtensor(h_hat[0], h[0])
        #p = T.nnet.sigmoid(T.dot(h_hat, HX) + bx)
        #energy = self.energy(x, p)
        #grads = theano.grad(energy, wrt=h_hat, consider_constant=[x])
        #return (h_hat - l * grads).astype(floatX)

    def move_h(self, h, x, l, HX, bx, *params):
        h0 = h[0]
        h1 = self.step_h(x[0], h0, *params)
        h = T.concatenate([h0[None, :, :], h1[None, :, :]], axis=0)
        p = T.nnet.sigmoid(T.dot(h, HX) + bx)
        energy = self.energy(x, p).mean()
        grad = theano.grad(energy, wrt=h0, consider_constant=[x])
        h0 = h0 - l * grad
        h1 = self.step_h(x[0], h0, *params)
        h = T.concatenate([h0[None, :, :], h1[None, :, :]], axis=0)
        return h

    def move_h_single(self, h, x, l, HX, bx):
        p = T.nnet.sigmoid(T.dot(h, HX) + bx)
        energy = self.energy_single(x, p)
        grads = theano.grad(energy, wrt=h, consider_constant=[x])
        return (h - l * grads).astype(floatX), p

    def step_infer(self, h, m, x, l, HX, bx, *params):
        def _shift_right(y):
            y_s = T.zeros_like(y)
            y_s = T.set_subtensor(y_s[1:], y[:-1])
            return y

        #h_ = _shift_right(h)
        #x_ = _shift_right(x)
        #h = self.move_h(x_, x, h_, h, l, HX, bx, *params)
        h = self.move_h(h, x, l, HX, bx, *params)
        p = T.nnet.sigmoid(T.dot(h, HX) + bx)
        x_hat = self.trng.binomial(p=p, size=p.shape, n=1, dtype=p.dtype)
        #x = m * x + (1. - m) * x_hat
        energy = self.energy(x, p)

        return h, x_hat, p, energy

    def inference(self, x, mask, l, n_inference_steps=1):
        x0 = x[0]
        p0 = T.zeros_like(x0) + x0
        #h0 = T.log(T.dot(T.log(p0 / (1 - p0 + 1e-7) + 1e-7) - self.bx, self.HX.T))
        h0 = T.alloc(0., x0.shape[0], self.dim_h).astype(floatX) + self.h0[None, :]

        seqs = [mask[1:], x[1:]]
        outputs_info = [x0, p0, h0]
        non_seqs = self.get_non_seqs()

        n_steps = x.shape[0] - 1
        (x1, p, h), updates = theano.scan(
            self.step_slice,
            sequences=seqs,
            outputs_info=outputs_info,
            non_sequences=non_seqs,
            name=tools._p(self.name, 'rnn_ff'),
            n_steps=n_steps,
            profile=tools.profile,
            strict=True
        )
        x = T.concatenate([x0[None, :, :], x1], axis=0)
        h = T.concatenate([h0[None, :, :], h], axis=0)
        updates = theano.OrderedUpdates(updates)
        seqs = []
        outputs_info = [h, None, None, None]
        x0_n = x0 * self.trng.binomial(p=0.5, size=x0.shape, n=1, dtype=x0.dtype)
        x_n = T.concatenate([x0_n[None, :, :], x1], axis=0)
        non_seqs = [mask, x_n, l, self.HX, self.bx] + self.get_non_seqs()

        (hs, x_hats, ps, energies), updates_2 = theano.scan(
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

        energy = self.energy(x, ps[-1])
        h0_mean = hs[-1].mean(axis=(0, 1))
        new_h = (1. - self.rate) * h0_mean + self.rate * h0_mean
        updates += [(self.h0, new_h)]
        return (x_hats, energy), updates

    def step_sample(self, x_, p_, h_, XHa, Ura, bha, XHb, Urb, bhb, HX, bx):
        preact = T.dot(h_, Ura) + T.dot(x_, XHa) + bha
        r, u = self.get_gates(preact)
        preactx = T.dot(h_, Urb) * r + T.dot(x_, XHb) + bhb
        h = T.tanh(preactx)
        h = u * h_ + (1. - u) * h
        p = T.nnet.sigmoid(T.dot(h, HX) + bx)
        x = self.trng.binomial(p=p, size=p.shape, n=1, dtype=p.dtype)
        return x, p, h

    def sample(self, x0=None, l=0.1, n_steps=100, n_samples=1,
               n_inference_steps=1000):
        if x0 is None:
            x0 = self.trng.binomial(p=0.5, size=(n_samples, self.dim_in), n=1,
                                    dtype=floatX)
        h0 = T.alloc(0., x0.shape[0], self.dim_h).astype(floatX) + self.h0[None, :]

        seqs = []
        outputs_info = [h0, None]
        non_seqs = [x0, l, self.HX, self.bx]

        (hs, ps), updates = theano.scan(
            self.move_h_single,
            sequences=seqs,
            outputs_info=outputs_info,
            non_sequences=non_seqs,
            name=tools._p(self.name, 'sample_init'),
            n_steps=n_inference_steps,
            profile=tools.profile,
            strict=True
        )

        h0 = hs[-1]
        p0 = ps[-1]

        seqs = []
        outputs_info = [x0, p0, h0]
        non_seqs = self.get_non_seqs()

        (xs, ps, hs), updates_2 = theano.scan(
            self.step_sample,
            sequences=seqs,
            outputs_info=outputs_info,
            non_sequences=non_seqs,
            name=tools._p(self.name, 'sample_chains'),
            n_steps=n_steps,
            profile=tools.profile,
            strict=True
        )
        updates.update(updates_2)

        return (xs, ps), updates

class CondGenGRU(GenerativeGRU):
    def __init__(self, dim_in, dim_h, weight_noise=False, name='cond_gen_gru',
                 trng=None, stochastic=True):
        self.stochastic = stochastic
        if trng is None:
            self.trng = RandomStreams(6 * 10 * 2015)
        else:
            self.trng = trng
        super(CondGenGRU, self).__init__(dim_in, dim_h,
                                         weight_noise=weight_noise, name=name)

    def set_params(self):
        norm_weight = tools.norm_weight
        super(CondGenGRU, self).set_params()
        Wx0a = np.concatenate([norm_weight(self.dim_in, self.dim_h),
                               norm_weight(self.dim_in, self.dim_h)], axis=1)
        Wx0b = norm_weight(self.dim_in, self.dim_h)
        self.params.update(Wx0a=Wx0a, Wx0b=Wx0b)

    def step_energy(self, x, q, e_):
        e = (x * T.log(q + 1e-7) + (1. - x) * T.log(1. - q + 1e-7)).sum(axis=1)
        return e_ + e, e

    def energy(self, x, q):
        n_steps = x.shape[0] - 1
        x = x[1:]
        q = q[1:]
        seqs = [x, q]
        outputs_info = [T.alloc(0., x.shape[1]).astype(floatX), None]
        non_seqs = []

        rval, updates = rval, updates = theano.scan(
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

    def __call__(self, x0, xT, reverse=True, n_steps=10):
        if reverse:
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

        if reverse:
            x = x[::-1]
            p = p[::-1]
            h = h[::-1]

        return OrderedDict(x=x, p=p, h=h, x0=x0, p0=p0, h0=h0, xT=xT), updates
