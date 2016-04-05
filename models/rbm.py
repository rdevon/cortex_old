'''
Module for RBM class
'''

from collections import OrderedDict
import numpy as np
import theano
from theano import tensor as T

from . import Layer
from utils import floatX
from utils.tools import (
    init_rngs,
    init_weights,
    norm_weight,
    ortho_weight,
    scan
)

def unpack(dim_h=None,
           dim_in=None,
           **model_args):

    dim_in = int(dim_in)
    dim_h = int(dim_h)

    rbm = RBM(dim_in, dim_h)
    models = [rbm]

    return models, model_args, None


class RBM(Layer):
    def __init__(self, dim_v, dim_h, name='rbm', **kwargs):

        self.dim_v = dim_v
        self.dim_h = dim_h

        self.h_act = 'T.nnet.sigmoid'
        self.v_act = 'T.nnet.sigmoid'

        kwargs = init_weights(self, **kwargs)
        kwargs = init_rngs(self, **kwargs)
        super(RNN, self).__init__(name=name, **kwargs)

    def set_params(self):
        W = norm_weight(self.dim_v, self.dim_h)
        b = np.zeros((self.dim_v,)).astype(floatX)
        c = np.zeros((self.dim_h,)).astype(floatX)

        self.params = OrderedDict(W=W, b=b, c=c)

    def get_params(self):
        return [self.W, self.b, self.c]

    def step_pv_h(self, h, W, b):
        '''Step function for probility of v given h'''
        return eval(self.v_act)(T.dot(h, W.T) + b)

    def step_sv_h(self, r, h, W, b):
        '''Step function for samples from v given h'''
        p = self.step_pv_h(h, W, b)
        return (r <= p).astype(floatX), p

    def step_ph_v(self, x, W, c):
        '''Step function for probability of h given v'''
        z = T.dot(x, W) + c
        p = eval(self.h_act)(z)
        return p

    def step_sh_v(self, r, x, W, c):
        '''Step function for sampling h given v'''
        p = self.step_ph_v(x, W, c)
        return (r <= p).astype(floatX), p

    def step_gibbs(self, r_v, r_h, x, W, b, c):
        '''Step Gibbs sample'''
        ph, h = self.step_sh_v(r_h, x, W, c)
        pv, v = self.step_sv_h(r_v, h, W, b)
        return v, h, pv, ph

    def sample(self, x0=None, h0=None, n_steps=1, n_samples=10):
        '''Gibbs sampling function.'''

        if x0 is None:
            assert h0 is not None
            r = self.trng.uniform(size=(h0.shape[0], n_samples))
            x0 = self.step_sv_h(r, h0[:, None, :], self.W, self.b)
        else:
            x0 = T.zeros((x0.shape[0], n_samples, self.dim_v)) + x0[:, None, :]

        r_vs = self.trng.uniform(
            size=(n_samples, x0.shape[0], x0.shape[1]), dtype=floatX)

        r_hs = self.trng.uniform(
            size=(n_samples, x0.shape[0], x0.shape[1]), dtype=floatX)

        seqs = [r_vs, r_hs]
        outputs_info = [x0, None, None, None]
        non_seqs = self.get_params()

        (vs, hs, pvs, pxs), updates = scan(
            self._step_sample, seqs, outputs_info, non_seqs, n_steps,
            name=self.name+'_sample', strict=False)

        return OrderedDict(x=x, h=h, p=p, q=q), updates

    def energy(self, v, h):
        '''Energy of a visible, hidden configuration.'''
        joint_term = -(h[:, None, :] * W[None, :, :] * v[:, :, None]).sum(axis=(1, 2))
        v_bias_term = (v * self.b[None, :]).sum(axis=1)
        h_bias_term = (h * self.c[None, :]).sum(axis=1)
        energy = -joint_term - v_bias_term - h_bias_term
        return energy

    def __call__(self, x):
        n_steps = x.shape[0]
        n_samples = x.shape[1]

        y, h, p, q = self._step(x, *self.get_params())

        return OrderedDict(y=y, h=h, p=p, q=q), theano.OrderedUpdates()


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