'''
Module for gradient inference methods.
These are instantiated as classes, possibly for good reasons. Possibly not.
Note that Theano will not accept any classes in scan. These rather push and pull the correct hyperparameters
 and parameters for use in scan.
'''

import theano
from theano import tensor as T


floatX = 'float32'

def load_inference(n_params, method=None, **kwargs):
    if method == 'sgd':
        return SGD(n_params, **kwargs)
    elif method == 'momentum':
        return Momentum(n_params, **kwargs)
    else:
        raise ValueError(method)


class SGD(object):
    def __init__(self, n_params, n_inference_steps=None, inference_rates=None, decay=None):
        self.n_params = n_params
        self.inference_rates = inference_rates
        self.decay = decay
        self.n_inference_steps = T.constant(n_inference_steps).astype('int64')

    def init_inference(self, *args):
        return self.inference_rates

    def get_params(self):
        return []

    def step(self, *args):
        model = self.model
        zs = model.get_qparams(*args)
        ls = model.get_qrest(*args)
        cost, grads = model.inference_cost(*args)
        zs = [(z - l * grad).astype(floatX) for z, l, grad in zip(zs, ls, grads)]
        if self.decay is not None:
            ls = [l * self.decay for l in ls]
        return tuple(zs + ls + [cost])

    def unpack(self, outs):
        zs = outs[:self.n_params]
        cost = outs[-1]
        return zs, cost

    def get_steps(self):
        return self.n_inference_steps, theano.OrderedUpdates()


class Momentum(object):
    def __init__(self, n_params, n_inference_steps=None, inference_rates=None,
                 momentum=None, decay=None):
        self.n_params = n_params
        self.inference_rates = inference_rates
        self.decay = decay
        assert len(inference_rates) == n_params
        self.n_inference_steps = T.constant(n_inference_steps).astype('int64')
        self.momentum = momentum

    def init_inference(self, *args):
        zs = args[-1]
        return self.inference_rates + [T.zeros_like(z) for z in zs]

    def get_params(self):
        return [T.constant(self.momentum).astype('float32')]

    def step(self, *args):
        model = self.model
        zs = model.get_qparams(*args)
        rest = model.get_qrest(*args)
        m = rest[-1]
        rest = rest[:-1]
        ls = rest[:len(rest) // 2]
        dzs = rest[len(rest) // 2:]
        cost, grads = model.inference_cost(*args)
        dzs = [(-l * grad + m * dz).astype(floatX)
            for l, grad, dz in zip(ls, grads, dzs)]
        zs = [(z + dz).astype(floatX) for z, dz in zip(zs, dzs)]
        if self.decay is not None:
            ls = [l * self.decay for l in ls]
        return tuple(zs + ls + dzs + [cost])

    def unpack(self, outs):
        zs = outs[:self.n_params]
        cost = outs[-1]
        return zs, cost

    def get_steps(self):
        return self.n_inference_steps, theano.OrderedUpdates()





# ADAM
def _step_adam(self, ph, preact_h1, y, z1, z2, m1_tm1, v1_tm1, m2_tm1, v2_tm1, cnt, b1, b2, lr, *params):

    b1 = b1 * (1 - 1e-7)**cnt
    cost, (grad1, grad2) = self.inference_cost(ph, preact_h1, y, z1, z2, *params)
    m1_t = b1 * m1_tm1 + (1 - b1) * grad1
    v1_t = b2 * v1_tm1 + (1 - b2) * grad1**2
    m2_t = b1 * m2_tm1 + (1 - b1) * grad2
    v2_t = b2 * v2_tm1 + (1 - b2) * grad2**2
    m1_t_hat = m1_t / (1. - b1**(cnt + 1))
    v1_t_hat = v1_t / (1. - b2**(cnt + 1))
    m2_t_hat = m2_t / (1. - b1**(cnt + 1))
    v2_t_hat = v2_t / (1. - b2**(cnt + 1))
    grad1_t = m1_t_hat / (T.sqrt(v1_t_hat) + 1e-7)
    grad2_t = m2_t_hat / (T.sqrt(v2_t_hat) + 1e-7)
    z1_t = (z1 - lr * grad1_t).astype(floatX)
    z2_t = (z2 - lr * grad2_t).astype(floatX)
    cnt += 1

    return z1_t, z2_t, m1_t, v1_t, m2_t, v2_t, cnt, cost

def _init_adam(self, ph, y, z):
    return [T.zeros_like(z), T.zeros_like(z), T.zeros_like(z), T.zeros_like(z), 0]

def _unpack_adam(self, outs):
    z1s, z2s, m1s, v1s, m2s, v2s, cnts, costs = outs
    return z1s, z2s, costs

def _inference_cost_cg(self, ph1, preact_h1, y, z1, z2,  *params):
    mu1 = eval(self.cond_to_h1.out_act)(preact_h1 + z1)
    #mu1 = T.nnet.sigmoid(z1)
    ph2 = self.p_h2_given_h1(mu1, *params)

    h1 = self.cond_to_h1.sample(mu1)

    mu2 = eval(self.cond_to_h2.out_act)(
        self.preact_h1_to_h2(h1, *params) + z2)
    py = self.p_y_given_h2(mu2, *params)

    cost = (self.cond_to_h1.neg_log_prob(mu1, ph1)
            + self.cond_to_h2.neg_log_prob(mu2, ph2)
            + self.cond_from_h2.neg_log_prob(y, py)
            - self.cond_to_h1.entropy(mu1)
            - self.cond_to_h2.entropy(mu2)
            )
    return cost

 # Conjugate gradient with line search
def _step_cg(self, ph1, preact_h1, y, z1, z2, s_, dz_sq_, alphas, *params):
    cost, (grad1, grad2) = self.inference_cost(ph1, preact_h1, y, z1, z2, *params)
    grad = T.concatenate([grad1, grad2], axis=1)
    dz = -grad
    dz_sq = (dz * dz).sum(axis=1)
    beta = dz_sq / (dz_sq_ + 1e-8)
    s = dz + beta[:, None] * s_

    z1_alpha = z1[None, :, :] + alphas[:, None, None] * s[None, :, :z1.shape[1]]
    z2_alpha = z2[None, :, :] + alphas[:, None, None] * s[None, :, z1.shape[1]:]
    costs = self._inference_cost_cg(
        ph1[None, :, :], preact_h1[None, :, :], y[None, :, :], z1_alpha, z2_alpha, *params)
    idx = costs.argmin(axis=0)
    z1 = z1 + alphas[idx][:, None] * s[:, :z1.shape[1]]
    z2 = z2 + alphas[idx][:, None] * s[:, z1.shape[1]:]
    return z1, z2, s, dz_sq, cost

def _init_cg(self, ph, y, z):
    params = self.get_params()
    s0 = T.alloc(0., z.shape[0], 2 * z.shape[1]).astype(floatX)
    dz_sq0 = T.alloc(1., z.shape[0]).astype(floatX)
    return [s0, dz_sq0]

def _unpack_cg(self, outs):
    z1s, z2s, ss, dz_sqs, costs = outs
    return z1s, z2s, costs

# Separate CG for z1 and z2
def _step_cg2(self, ph1, preact_h1, y, z1, z2, s1_, s2_, dz1_sq_, dz2_sq_, alphas, *params):
    cost, (grad1, grad2) = self.inference_cost(ph1, preact_h1, y, z1, z2, *params)
    dz1 = -grad1
    dz2 = -grad2
    dz1_sq = (dz1 * dz1).sum(axis=1)
    dz2_sq = (dz2 * dz2).sum(axis=1)
    beta1 = dz1_sq / (dz1_sq_ + 1e-8)
    beta2 = dz2_sq / (dz2_sq_ + 1e-8)
    s1 = dz1 + beta1[:, None] * s1_
    s2 = dz2 + beta2[:, None] * s2_

    z1_alpha = z1[None, :, :] + alphas[:, None, None] * s1[None, :, :]
    z2_alpha = z2[None, :, :] + alphas[:, None, None] * s2[None, :, :]

    z1_e = T.alloc(0., alphas.shape[0], z1.shape[0], z1.shape[1]).astype(floatX) + z1
    z2_e = T.alloc(0., alphas.shape[0], z2.shape[0], z2.shape[1]).astype(floatX) + z2

    costs2 = self._inference_cost_cg(
        ph1[None, :, :], preact_h1[None, :, :], y[None, :, :], z1_e, z2_alpha, *params)

    costs1 = self._inference_cost_cg(
        ph1[None, :, :], preact_h1[None, :, :], y[None, :, :], z1_alpha, z2_e, *params)

    idx2 = costs2.argmin(axis=0)
    z2 = z2 + alphas[idx2][:, None] * s2
    idx1 = costs1.argmin(axis=0)
    z1 = z1 + alphas[idx1][:, None] * s1
    return z1, z2, s1, s2, dz1_sq, dz2_sq, cost

def _init_cg2(self, ph, y, z):
    params = self.get_params()
    s10 = T.zeros_like(z)
    s20 = T.zeros_like(z)
    dz1_sq0 = T.alloc(1., z.shape[0]).astype(floatX)
    dz2_sq0 = T.alloc(1., z.shape[0]).astype(floatX)
    return [s10, s20, dz1_sq0, dz2_sq0]

def _unpack_cg2(self, outs):
    z1s, z2s, s1s, s2s, dz1_sqs, dz2_sqs, costs = outs
    return z1s, z2s, costs

def _params_cg2(self):
    return [(self.inference_rate_1 * 2. ** T.arange(9)).astype(floatX)]