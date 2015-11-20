"""
Module for GRU layers
"""

import copy
from collections import OrderedDict
import numpy as np
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from layers import FFN
from layers import MLP
from layers import Layer
from rnn import RNN
from rnn import GenRNN
import tools
from tools import gaussian
from tools import log_gaussian
from tools import init_rngs
from tools import init_weights


norm_weight = tools.norm_weight
ortho_weight = tools.ortho_weight
floatX = 'float32'#theano.config.floatX

def raise_type_error(o, t):
    raise ValueError('%s is not of type %s' % (type(o), t))

pi = theano.shared(np.pi).astype(floatX)


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
    def __init__(self, dim_in, dim_h, weight_noise=False, weight_scale=0.01,
                 learn_h0=False, name='gru', rng=None):
        self.weight_noise = weight_noise
        self.weight_scale = weight_scale
        self.learn_h0 = learn_h0

        if rng is None:
            rng = tools.rng_
        self.rng = rng

        super(GRU, self).__init__(dim_in, dim_h, name=name)
        self.set_params()

    def set_params(self):
        norm_weight = tools.norm_weight
        ortho_weight = tools.ortho_weight
        W = np.concatenate([norm_weight(self.dim_in, self.dim_h,
                                        scale=self.weight_scale,
                                        rng=self.rng),
                            norm_weight(self.dim_in, self.dim_h,
                                        scale=self.weight_scale,
                                        rng=self.rng)], axis=1)
        b = np.zeros((2 * self.dim_h,)).astype(floatX)
        U = np.concatenate([ortho_weight(self.dim_h),
                            ortho_weight(self.dim_h)], axis=1)
        Wx = norm_weight(self.dim_in, self.dim_h, scale=self.weight_scale,
                         rng=self.rng)
        Ux = ortho_weight(self.dim_h, rng=self.rng)
        bx = np.zeros((self.dim_h,)).astype(floatX)
        self.params = OrderedDict(W=W, b=b, U=U, Wx=Wx, Ux=Ux, bx=bx)

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
    def __init__(self, dim_in, dim_h, dim_o, window=1, convolve_time=False,
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

    def step_slice(self, m_, x_, xx_, h_, U, Ux, Wo, bo):
        preact = self.recurrent_step(T.dot(h_, U), x_)
        r, u = self.get_gates(preact)
        preactx = T.dot(h_, Ux) * r + xx_
        h = T.tanh(preactx)
        h = u * h_ + (1. - u) * h
        h = m_[:, None] * h + (1 - m_)[:, None] * h_
        o = T.dot(h, Wo) + bo
        return h, o

    def __call__(self, state_below, mask=None, suppress_noise=False):
        n_steps = state_below.shape[0]
        n_samples = state_below.shape[1]

        x, x_ = self.set_inputs(state_below, suppress_noise=suppress_noise)
        if mask is None:
            mask = T.alloc(1., state_below.shape[0], state_below.shape[1])

        if self.weight_noise and not suppress_noise:
            U = self.U + self.U_noise
            Ux = self.Ux + self.Ux_noise
            Wo = self.Wo + self.Wo_noise
        else:
            U = self.U
            Ux = self.Ux
            Wo = self.Wo

        seqs = [mask, x, x_]
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

        return OrderedDict(h=h, o=out), updates


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

    def __call__(self, state_below, mask_f=None, suppress_noise=False):
        n_steps = state_below.shape[0]
        n_samples = state_below.shape[1]

        mask = T.neq(state_below[:, :, 0], 1).astype(floatX)
        if mask_f is not None:
            mask = mask * mask_f
        mask_r = T.roll(mask, 1, axis=0)
        mask_n = T.eq(mask, 0.).astype(floatX)
        if mask_f is not None:
            mask_n = mask_n * mask_f

        x, x_ = self.set_inputs(state_below, suppress_noise=suppress_noise)

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

    def __call__(self, state_below, mask_f=None, suppress_noise=False):
        n_steps = state_below.shape[0]
        n_samples = state_below.shape[1]

        mask = T.neq(state_below[:, :, 0], 1).astype(floatX)
        if mask_f is not None:
            mask = mask * mask_f
        mask_r = T.roll(mask, 1, axis=0)
        mask_n = T.eq(mask, 0.).astype(floatX)
        if mask_f is not None:
            mask_n = mask_n * mask_f

        x, x_ = self.set_inputs(state_below)


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

        def step_compress(mask, out):
            n_outs = mask.sum().astype('int64')
            return mask.compress(out, axis=0).flatten()

        o, _ = theano.scan(
            step_compress,
            sequences=[mask_n[1:].T, o.dimshuffle(1, 0, 2)],
            outputs_info=[None],
            strict=True
        )

        return OrderedDict(h=h, o=o, mask=mask, mask_n=mask_n), updates


class GenerativeGRU(GRU):
    def __init__(self, dim_in, dim_h, weight_noise=False, name='gen_gru',
                 trng=None, condition_on_x=False):
        if weight_noise:
            raise NotImplementedError()

        if trng is None:
            self.trng = RandomStreams(6 * 10 * 2015)
        else:
            self.trng = trng

        self.condition_on_x = condition_on_x

        self.weight_noise = weight_noise
        super(GenerativeGRU, self).__init__(dim_in, dim_h, name=name)

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


class GenGRU(Layer):
    def __init__(self, dim_in, dim_h,
                 input_net=None, input_net_aux=None, output_net=None, conditional=None,
                 f_sample=None, f_neg_log_prob=None,
                 name='gen_gru', **kwargs):

        self.dim_in = dim_in
        self.dim_h = dim_h

        self.input_net = input_net
        self.input_net_aux = input_net_aux
        self.output_net = output_net
        self.conditional = conditional

        self.f_sample = f_sample
        self.f_neg_log_prob = f_neg_log_prob

        kwargs = init_weights(self, **kwargs)
        kwargs = init_rngs(self, **kwargs)

        assert len(kwargs) == 0, kwargs.keys()
        super(GenGRU, self).__init__(name=name)

        if self.weight_noise:
            raise NotImplementedError()

    def get_gates(self, preact):
        r = T.nnet.sigmoid(RNN._slice(preact, 0, self.dim_h))
        u = T.nnet.sigmoid(RNN._slice(preact, 1, self.dim_h))
        return r, u

    def set_params(self):
        Ura = np.concatenate([ortho_weight(self.dim_h),
                              ortho_weight(self.dim_h)], axis=1)
        Urb = ortho_weight(self.dim_h)

        self.params = OrderedDict(Ura=Ura, Urb=Urb)

        if self.input_net is None:
            self.input_net = MLP(self.dim_in, self.dim_h, self.dim_h, 1,
                            rng=self.rng, trng=self.trng,
                            h_act='T.nnet.sigmoid',
                            out_act='T.tanh',
                            name='input_net')
        else:
            assert self.input_net.dim_in == self.dim_in
            assert self.input_net.dim_out == self.dim_h
            self.input_net.name = 'input_net'

        if self.input_net_aux is None:
            self.input_net_aux = MLP(self.dim_in, 2 * self.dim_h, 2 * self.dim_h, 1,
                            rng=self.rng, trng=self.trng,
                            h_act='T.nnet.sigmoid',
                            out_act='T.tanh',
                            name='input_net_aux')
        else:
            assert self.input_net_aux.dim_in == self.dim_in
            assert self.input_net_aux.dim_out == 2 * self.dim_h
            self.input_net_aux.name = 'input_net_aux'

        if self.output_net is None:
            self.output_net = MLP(self.dim_h, self.dim_h, self.dim_in, 1,
                            rng=self.rng, trng=self.trng,
                            f_sample=self.f_sample,
                            f_neg_log_prob=self.f_neg_log_prob,
                            h_act='T.nnet.sigmoid',
                            out_act='T.nnet.sigmoid',
                            name='output_net')
        else:
            assert self.output_net.dim_in == self.dim_h
            self.output_net.name = 'output_net'

        if self.conditional is not None:
            assert self.conditional.dim_in == self.dim_in
            assert self.conditional.dim_out == self.dim_in
            self.conditional.name = 'conditional'

    def set_tparams(self):
        tparams = super(GenGRU, self).set_tparams()
        for mlp in [self.input_net_aux, self.input_net, self.output_net]:
            tparams.update(**mlp.set_tparams())

        if self.conditional is not None:
            tparams.update(**self.conditional.set_tparams())
        return tparams

    def get_sample_params(self):
        params = [self.Ura, self.Urb]
        params += self.input_net_aux.get_params()
        params += self.input_net.get_params()
        params += self.output_net.get_params()
        if self.conditional is not None:
            params += self.conditional.get_params()
        return params

    def get_params(self):
        params = [self.Ura, self.Urb]
        return params

    def get_recurrent_args(self, *args):
        return args[:2]

    def get_aux_args(self, *args):
        start = 2
        length = len(self.input_net_aux.get_params())
        return args[start:start+length]

    def get_input_args(self, *args):
        start = 2 + len(self.input_net.get_params())
        length = len(self.input_net.get_params())
        return args[start:start+length]

    def get_output_args(self, *args):
        start = 2 + len(self.input_net_aux.get_params()) + len(self.input_net.get_params())
        length = len(self.output_net.get_params())
        return args[start:start+length]

    def get_conditional_args(self, *args):
        start = 2 + len(self.input_net_aux.get_params()) + len(self.input_net.get_params()) + len(self.output_net.get_params())
        length = len(self.conditional.get_params())
        return args[start:start+length]

    def step_sample(self, h_, x_, *params):
        Ura, Urb = self.get_recurrent_args(*params)

        aux_params = self.get_aux_args(*params)
        input_params = self.get_input_args(*params)
        output_params = self.get_output_args(*params)

        y_aux = self.input_net_aux.preact(x_, *aux_params)
        y_input = self.input_net.preact(x_, *input_params)

        h = self._step(y_aux, y_input, h_, Ura, Urb)

        preact = self.output_net.preact(h, *output_params)
        if self.conditional is not None:
            c_params = self.get_conditional_args(*params)
            preact += self.conditional.preact(x_, *c_params)

        p = eval(self.output_net.out_act)(preact)
        x = self.output_net.sample(p)
        return h, x, p

    def _step(self, y_a, y_i, h_, Ura, Urb):
        preact = T.dot(h_, Ura) + y_a
        r, u = self.get_gates(preact)
        preactx = T.dot(h_, Urb) * r + y_i
        h = T.tanh(preactx)
        h = u * h + (1. - u) * h_
        return h

    def _energy(self, X, h0=None):
        outs, updates = self.__call__(X[:-1], h0=h0)
        p = outs['p']
        energy = self.output_net.net_log_prob(X[1:], p).sum(axis=0)
        return energy

    def sample(self, x0=None, h0=None, n_samples=10, n_steps=10):
        if x0 is None:
            x0 = self.output_net.sample(
                p=0.5, size=(n_samples, self.output_net.dim_out)).astype(floatX)
        else:
            x0 = self.output_net.sample(x0)

        p0 = x0.copy()
        if h0 is None:
            h0 = T.alloc(0., x0.shape[0], self.dim_h).astype(floatX)

        seqs = []
        outputs_info = [h0, x0, None]
        non_seqs = self.get_sample_params()

        (h, x, p), updates = theano.scan(
            self.step_sample,
            sequences=seqs,
            outputs_info=outputs_info,
            non_sequences=non_seqs,
            name=tools._p(self.name, '_sampling'),
            n_steps=n_steps,
            profile=tools.profile,
            strict=True)

        x = tools.concatenate([x0[None, :, :], x])
        h = tools.concatenate([h0[None, :, :], h])
        p = tools.concatenate([p0[None, :, :], p])

        return OrderedDict(x=x, p=p, h=h, x0=x0, p0=p0, h0=h0), updates

    def __call__(self, x, h0=None):
        n_steps = x.shape[0]
        n_samples = x.shape[1]

        if h0 is None:
            h0 = T.alloc(0., n_samples, self.dim_h).astype(floatX)

        a = self.input_net_aux(x, return_preact=True)
        b = self.input_net(x, return_preact=True)
        seqs = [a, b]
        outputs_info = [h0]
        non_seqs = self.get_params()

        h, updates = theano.scan(
            self._step,
            sequences=seqs,
            outputs_info=outputs_info,
            non_sequences=non_seqs,
            name=tools._p(self.name, '_recurrent_steps'),
            n_steps=n_steps,
            profile=tools.profile,
            strict=True)

        preact = self.output_net(h, return_preact=True)
        if self.conditional is not None:
            preact += self.conditional(x, return_preact=True)
        p = eval(self.output_net.out_act)(preact)
        y = self.output_net.sample(p=p)

        return OrderedDict(h=h, y=y, p=p, a=a, b=b), updates

class MultiLayerGRU(Layer):
    def __init__(self, dim_in, dim_h, n_layers=2,
                 weight_noise=False, weight_scale=0.01,
                 condition_on_x=None,
                 rng=None, trng=None,
                 name='gen_gru'):
        if weight_noise:
            raise NotImplementedError()

        self.dim_in = dim_in
        self.dim_h = dim_h
        self.n_layers = n_layers
        self.condition_on_x = condition_on_x

        self.weight_noise = weight_noise
        self.weight_scale = weight_scale

        if rng is None:
            rng = tools.rng_
        self.rng = rng

        if trng is None:
            self.trng = RandomStreams(random.randint(0, 10000))
        else:
            self.trng = trng

        super(MultiLayerGenGRU, self).__init__(name=name)

        self.layers = []
        for l in xrange(n_layers):
            if l == 0:
                dim_in = self.dim_in
            else:
                dim_in = self.dim_h + dim_in
            rnn = GenGru(dim_in, dim_h, name='gen_gru_%d' % l)
            self.layers.append(rnn)

    def set_params(self):
        for rnn in self.layers:
            rnn.set_params()

    def set_tparams(self):
        tparams = OrderedDict()
        for rnn in self.layers:
            tparams.update(**rnn.set_tparams())
        return tparams

    def get_params(self):
        params = []
        for rnn in self.layers:
            params += rnn.get_params()
        return params

    def get_params_level(self, level, *params):
        start = sum([0] + [len(rnn.get_params()) for rnn in self.layers[:level]])
        length = len(self.layers[level].get_params())
        return params[start:start+length]

    def _step(self, x, h_, l, *params):
        ps = self.get_params_level(l, *params)

    def __call__(self, x, h0s=None):
        n_steps = x.shape[0]
        n_samples = x.shape[1]

        if h0 is None:
            h0 = T.alloc(0., n_samples, self.dim_h).astype(floatX)

        seqs = [x]
        outputs_info = [h0]
        non_seqs = [self.XHa, self.Ura, self.bha, self.XHb, self.Urb, self.bhb]

        h, updates = theano.scan(
            self._step,
            sequences=seqs,
            outputs_info=outputs_info,
            non_sequences=non_seqs,
            name=tools._p(self.name, '_layers'),
            n_steps=n_steps,
            profile=tools.profile,
            strict=True)

        preact = T.dot(h, self.HX) + self.bx
        if self.condition_on_x is not None:
            preact += self.condition_on_x(x, return_preact=True)
        p = T.nnet.sigmoid(preact)
        y = self.trng.binomial(p=p, size=p.shape, n=1, dtype=p.dtype)

        return OrderedDict(h=h, y=y, p=p), updates


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
                 h0_mode='average', weight_noise=False, trng=None,
                 stochastic=True, rate=0.1):
        self.stochastic = stochastic
        self.rate = rate
        self.h0_mode = h0_mode
        if trng is None:
            self.trng = RandomStreams(6 * 10 * 2015)
        else:
            self.trng = trng
        super(SimpleInferGRU, self).__init__(dim_in, dim_h,
                                             weight_noise=weight_noise,
                                             name=name)

    def set_params(self):
        super(SimpleInferGRU, self).set_params()

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

    def move_h(self, h0, x, l, HX, bx, *params):
        h1 = self.step_h(x[0], h0, *params)
        h = T.concatenate([h0[None, :, :], h1[None, :, :]], axis=0)
        p = T.nnet.sigmoid(T.dot(h, HX) + bx)
        energy = self.energy(x, p).mean()
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

    def inference(self, x, mask, l, n_inference_steps=1, max_k=10):
        # Initializing the variables
        x0 = x[0]
        x1 = x[1]
        #x0_n = x0 * self.trng.binomial(p=0.1, size=x0.shape, n=1, dtype=x0.dtype)
        #x_n = T.concatenate([x0_n[None, :, :], x1[None, :, :]], axis=0)

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
            new_h = (1. - self.rate) * self.h0 + self.rate * h0_mean
            updates += [(self.h0, new_h)]

        return (x_hats, h0s, energy), updates

    def step_sample(self, x_, p_, h_, XHa, Ura, bha, XHb, Urb, bhb, HX, bx):
        preact = T.dot(h_, Ura) + T.dot(x_, XHa) + bha
        r, u = self.get_gates(preact)
        preactx = T.dot(h_, Urb) * r + T.dot(x_, XHb) + bhb
        h = T.tanh(preactx)
        h = u * h_ + (1. - u) * h
        p = T.nnet.sigmoid(T.dot(h, HX) + bx)
        x = self.trng.binomial(p=p, size=p.shape, n=1, dtype=p.dtype)
        return x, p, h

    def sample(self, x0=None, x1=None, l=0.01, n_steps=100, n_samples=1,
               n_inference_steps=100):
        if x0 is None:
            x0 = self.trng.binomial(p=0.5, size=(n_samples, self.dim_in), n=1,
                                    dtype=floatX)

        if self.h0_mode == 'average':
            h0 = T.alloc(0., x0.shape[0], self.dim_h).astype(floatX) + self.h0[None, :]
        elif self.h0_mode == 'ffn':
            if x1 is None:
                x1 = x0
            h0 = T.dot(x0, self.W0) + T.dot(x1, self.U0) + self.b0

        p0 = T.nnet.sigmoid(T.dot(h0, self.HX) + self.bx)

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
