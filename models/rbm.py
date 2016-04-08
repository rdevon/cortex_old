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
    concatenate,
    init_rngs,
    init_weights,
    log_sum_exp,
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
    '''
    RBM class.

    Attributes:
        dim_v: int. number of visible units.
        dim_h: int. number of hidden units.
        h_act: str. Theano function name for hidden activation.
        v_act: str. Theano function name for visible activation.
        W: T.tensor. weights
        b: T.tensor. visible bias
        c: T.tensor. hidden bias.
    '''
    def __init__(self, dim_v, dim_h, name='rbm', **kwargs):
        '''Init method for RBM class.
        '''
        self.dim_v = dim_v
        self.dim_h = dim_h

        self.h_act = 'T.nnet.sigmoid'
        self.v_act = 'T.nnet.sigmoid'

        kwargs = init_weights(self, **kwargs)
        kwargs = init_rngs(self, **kwargs)
        super(RBM, self).__init__(name=name, **kwargs)

    @staticmethod
    def factory(dim_v=None, dim_h=None, **kwargs):
        return RBM(dim_v, dim_h, **kwargs)

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

    def step_gibbs(self, r_h, r_v, h, W, b, c):
        '''Step Gibbs sample'''
        v, pv = self.step_sv_h(r_v, h, W, b)
        h, ph = self.step_sh_v(r_h, v, W, c)
        return h, v, ph, pv

    def sample(self, h0, n_steps=1):
        '''Gibbs sampling function.'''

        r_vs = self.trng.uniform(
            size=(n_steps, h0.shape[0], self.dim_v),
            dtype=floatX)

        r_hs = self.trng.uniform(
            size=(n_steps, h0.shape[0], self.dim_h),
            dtype=floatX)

        seqs = [r_hs, r_vs]
        outputs_info = [h0, None, None, None]
        non_seqs = self.get_params()

        (hs, vs, phs, pvs), updates = scan(
            self.step_gibbs, seqs, outputs_info, non_seqs, n_steps,
            name=self.name+'_sample', strict=False)

        return OrderedDict(vs=vs, hs=hs, pvs=pvs, phs=phs), updates

    def reconstruct(self, x):
        r = self.trng.uniform(
            size=(x.shape[0], self.dim_h),
            dtype=floatX)

        h, ph = self.step_sh_v(r, x, self.W, self.c)
        pv = self.step_pv_h(h, self.W, self.b)

        return pv

    def estimate_nll(self, X, K=100):
        log_za, d_logz = self.ais(X, K)
        log_Z = log_za + d_logz
        return (self.free_energy(X) + log_Z).mean(), self.free_energy(X).mean(), log_Z, log_za, d_logz

    def step_gibbs_ais(self, r_h_a, r_h_b, r_v, v, beta,
                       W_a, b_a, c_a, W_b, b_b, c_b):
        '''Step Gibbs sample for AIS'''
        W_a = (1 - beta) * W_a
        b_a = (1 - beta) * b_a
        c_a = (1 - beta) * c_a

        W_b = beta * W_b
        b_b = beta * b_b
        c_b = beta * c_b

        h_a, _ = self.step_sh_v(r_h_a, v, W_a, c_a)
        h_b, _ = self.step_sh_v(r_h_b, v, W_b, c_b)

        pv_act = T.dot(h_a, W_a.T) + T.dot(h_b, W_b.T) + b_a + b_b
        pv = eval(self.v_act)(pv_act)
        v = (r_v <= pv).astype(floatX)

        return v

    def ais(self, X, K):
        '''Performs AIS to estimate the log of the partition function, Z.'''

        def free_energy(beta, W_a, b_a, c_a, W_b, b_b, c_b):
            fe_a = self.step_free_energy(X, 1 - beta, W_a, b_a, c_a)
            fe_b = self.step_free_energy(X, beta, W_b, b_b, c_b)
            fe = fe_a + fe_b
            return fe

        def step_anneal(r_h_a, r_h_b, r_v, k, log_w, x, *params):
            beta_ = ((k - 1) / (K - 1)).astype(floatX)
            beta  = (k / (K - 1)).astype(floatX)
            log_w += free_energy(beta_, *params) - free_energy(beta, *params)
            x = self.step_gibbs_ais(r_h_a, r_h_b, r_v, x, beta, *params)
            return log_w, x

        def get_log_z(log_w):
            log_wn = log_w - T.log(log_w.shape[0])
            log_z = log_sum_exp(log_wn, axis=0)
            return log_z

        r_vs = self.trng.uniform(
            size=(K + 1, X.shape[0], self.dim_v),
            dtype=floatX)

        r_hs_a = self.trng.uniform(
            size=(K + 1, X.shape[0], self.dim_h),
            dtype=floatX)

        r_hs_b = self.trng.uniform(
            size=(K + 1, X.shape[0], self.dim_h),
            dtype=floatX)

        W_a = T.zeros_like(self.W).astype(floatX)
        X_mean = T.clip(X.mean(axis=0), 1e-6, 1 - 1e-6)
        b_a = -T.log(1 / X_mean - 1)
        c_a = T.zeros_like(self.c).astype(floatX)

        params = [W_a, b_a, c_a] + self.get_params()

        x0 = self.step_gibbs_ais(r_hs_a[0], r_hs_b[0], r_vs[0], X, 0, *params)
        log_w0 = T.zeros((X.shape[0],)).astype(floatX)

        seqs         = [r_hs_a[1:], r_hs_b[1:], r_vs[1:], T.arange(1, K)]
        outputs_info = [log_w0, x0]
        non_seqs     = params

        (log_ws, xs), updates = scan(
            step_anneal, seqs, outputs_info, non_seqs, K - 1,
            name=self.name + '_ais', strict=False
        )

        log_wk = log_ws[-1]
        d_logz = get_log_z(log_wk)
        log_za = self.dim_h * T.log(2.).astype(floatX) + T.log(1. + T.exp(b_a)).sum(axis=0)

        return log_za, d_logz

    def energy(self, v, h):
        if v.ndim == 3:
            reduce_dims = (v.shape[0], v.shape[1])
            v = v.reshape((reduce_dims[0] * reduce_dims[1], v.shape[2]))
            h = h.reshape((reduce_dims[0] * reduce_dims[1], h.shape[2]))
        else:
            reduce_dims = None
        '''Energy of a visible, hidden configuration.'''
        joint_term = (h[:, None, :] * self.W[None, :, :] * v[:, :, None]).sum(axis=1)
        v_bias_term = (v * self.b[None, :]).sum(axis=1)
        h_bias_term = (h * self.c[None, :])
        energy = -joint_term - v_bias_term[:, None] - h_bias_term

        if reduce_dims is not None:
            energy = energy.reshape((reduce_dims[0], reduce_dims[1], energy.shape[1]))

        return energy

    def step_free_energy(self, x, beta, W, b, c):
        fe = -beta * (x * b[None, :]).sum(axis=1) - T.log(
            1 + T.exp(beta * (c[None, :] + T.dot(x, W)))).sum(axis=1)
        return fe

    def free_energy(self, x):
        if x.ndim == 3:
            reduce_dims = (x.shape[0], x.shape[1])
            x = x.reshape((reduce_dims[0] * reduce_dims[1], x.shape[2]))
        else:
            reduce_dims = None
        fe = self.step_free_energy(x, 1, *self.get_params())

        if reduce_dims is not None:
            fe = fe.reshape(reduce_dims)

        return fe

    def neg_log_prob(self, x, p):
        p = T.clip(p, 1e-6, 1 - 1e-6)
        return (-x * T.log(p) - (1 - x) * T.log(1 - p)).sum(axis=x.ndim-1)

    def __call__(self, x, h_p=None, n_steps=1, n_chains=10):
        '''Call function.

        Returns results, including generic cost function, and samples from
        Gibbs chain.

        '''
        ph0 = self.step_ph_v(x, self.W, self.c)
        if h_p is None:
            r = self.trng.uniform(size=(x.shape[0], self.dim_h))
            h_p = (r <= ph0).astype(floatX)
        outs, updates = self.sample(h0=h_p, n_steps=n_steps)

        v0 = x
        vk = outs['vs'][-1]

        positive_cost = self.free_energy(v0)
        negative_cost = self.free_energy(vk)
        cost          = positive_cost.mean() - negative_cost.mean()
        fe            = self.free_energy(v0)
        recon_error   = self.neg_log_prob(x, self.reconstruct(x)).mean()

        nll, fe_, log_z, log_za, d_logz = self.estimate_nll(x)
        results = OrderedDict(
            cost=cost,
            positive_cost=positive_cost.mean(),
            negative_cost=negative_cost.mean(),
            free_energy=fe.mean(),
            nll=nll,
            hid=T.log(1 + T.exp(self.c[None, :] + (self.W[None, :, :] * x[:, :, None]).sum(axis=1))).sum(axis=1).mean(),
            vis=(self.b[None, :] * x).sum(axis=1).mean(),
            log_z=log_z,
            log_za=log_za,
            logz_d=d_logz,
            recon_error=recon_error
        )

        samples = OrderedDict(
            vs=outs['vs'],
            hs=outs['hs'],
            pvs=outs['pvs'],
            phs=outs['phs'],
            positive_cost=positive_cost,
            negative_cost=negative_cost,
        )

        return results, samples, updates, []


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