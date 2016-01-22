'''
Module for Deep Sigmoid Belief Networks
'''

from collections import OrderedDict
import numpy as np
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from layers import Layer
from models.mlp import (
    MLP,
    MultiModalMLP
)
from models.sbn import (
    init_inference_args
)
from utils import tools
from utils.tools import (
    concatenate,
    floatX,
    init_rngs,
    init_weights,
    log_mean_exp,
    log_sum_exp,
    logit,
    norm_weight,
    ortho_weight,
    pi,
    _slice
)


class DeepSBN(Layer):
    def __init__(self, dim_in, dim_h=None, n_layers=2, dim_hs=None,
                 posteriors=None, conditionals=None,
                 z_init=None,
                 name='sbn',
                 **kwargs):

        self.dim_in = dim_in
        if dim_hs is not None:
            self.dim_hs = dim_hs
        else:
            assert dim_h is not None
            assert n_layers is not None
            self.dim_hs = [dim_h for _ in xrange(n_layers)]
        self.n_layers = len(self.dim_hs)

        self.posteriors = posteriors
        self.conditionals = conditionals

        self.z_init = z_init

        kwargs = init_inference_args(self, **kwargs)
        kwargs = init_weights(self, **kwargs)
        kwargs = init_rngs(self, **kwargs)

        super(DeepSBN, self).__init__(name=name)

    def set_params(self):
        z = np.zeros((self.dim_hs[-1],)).astype(floatX)

        self.params = OrderedDict(z=z)

        if self.posteriors is None:
            self.posteriors = [None for _ in xrange(self.n_layers)]
        else:
            assert len(self.posteriors) == self.n_layers

        if self.conditionals is None:
            self.conditionals = [None for _ in xrange(self.n_layers)]
        else:
            assert len(self.conditionals) == self.n_layers

        for l, dim_h in enumerate(self.dim_hs):
            if l == 0:
                dim_in = self.dim_in
            else:
                dim_in = self.dim_hs[l-1]

            if self.posteriors[l] is None:
                self.posteriors[l] = MLP(
                    dim_in, dim_h, dim_h, 2,
                    rng=self.rng, trng=self.trng,
                    h_act='T.nnet.softplus',
                    out_act='T.nnet.sigmoid')

            if self.conditionals[l] is None:
                self.conditionals[l] = MLP(
                    dim_h, dim_h, dim_in, 1,
                    rng=self.rng, trng=self.trng,
                    h_act='T.nnet.softplus',
                    out_act='T.nnet.sigmoid')

            if l == 0:
                self.posteriors[l].name = self.name + '_posterior'
                self.conditionals[l].name = self.name + '_conditional'
            else:
                self.posteriors[l].name = self.name + '_posterior%d' % l
                self.conditionals[l].name = self.name + '_conditional%d' % l

    def set_tparams(self, excludes=[]):
        excludes = ['{name}_{key}'.format(name=self.name, key=key)
                    for key in excludes]
        tparams = super(DeepSBN, self).set_tparams()

        for l in xrange(self.n_layers):
            tparams.update(**self.posteriors[l].set_tparams())
            tparams.update(**self.conditionals[l].set_tparams())

        tparams = OrderedDict((k, v) for k, v in tparams.iteritems()
            if k not in excludes)

        return tparams

    def get_params(self):
        params = [self.z]
        for l in xrange(self.n_layers):
            params += self.conditionals[l].get_params()
        return params

    def p_y_given_h(self, h, level, *params):
        start = 1
        for l in xrange(level):
            start += len(self.conditionals[l].get_params())
        end = start + len(self.conditionals[level].get_params())

        params = params[start:end]
        return self.conditionals[level].step_call(h, *params)

    def sample_from_prior(self, n_samples=100):
        p = T.nnet.sigmoid(self.z)
        h = self.posteriors[-1].sample(p=p, size=(n_samples, self.dim_hs[-1]))

        for conditional in self.conditionals[::-1]:
            p = conditional(h)
            h = conditional.sample(p)

        return p

    def kl_divergence(self, p, q):
        '''
        Negative KL divergence actually.
        '''
        p_c = T.clip(p, 1e-7, 1.0 - 1e-7)
        q = T.clip(q, 1e-7, 1.0 - 1e-7)

        entropy_term = T.nnet.binary_crossentropy(p_c, p)
        prior_term = T.nnet.binary_crossentropy(q, p)
        return (prior_term - entropy_term).sum(axis=entropy_term.ndim-1)

    def m_step(self, x, y, qks, n_samples=10):
        constants = []

        hs = []
        for l, qk in enumerate(qks):
            h = self.posteriors[l].sample(qk, size=(n_samples, qk.shape[0], qk.shape[1]))
            hs.append(h)
        p_ys = [conditional(h) for h, conditional in zip(hs, self.conditionals)]
        ys = [y[None, :, :]] + hs[:-1]

        #q0s = self.init_variational_params(x)

        q0s = []
        state = x
        for l in xrange(self.n_layers):
            q0 = self.posteriors[l](state)
            q0s.append(q0)
            state = qks[l]

        conditional_energy = T.constant(0.).astype(floatX)
        posterior_energy = T.constant(0.).astype(floatX)
        for l in xrange(self.n_layers):
            posterior_energy += self.posteriors[l].neg_log_prob(qks[l], q0s[l])
            conditional_energy += self.conditionals[l].neg_log_prob(
                ys[l], p_ys[l]).mean(axis=0)

        prior = T.nnet.sigmoid(self.z)
        prior_energy = self.posteriors[-1].neg_log_prob(qks[-1], prior[None, :])

        return (prior_energy.mean(axis=0), posterior_energy.mean(axis=0),
                conditional_energy.mean(axis=0)), constants

    def step_infer(self, *params):
        raise NotImplementedError()
    def init_infer(self, z):
        raise NotImplementedError()
    def unpack_infer(self, outs):
        raise NotImplementedError()
    def params_infer(self):
        raise NotImplementedError()

    # Importance Sampling
    def _step_adapt(self, y, *params):
        print 'AdIS'
        params = list(params)
        qs = params[:self.n_layers]
        params = params[self.n_layers:]
        prior = T.nnet.sigmoid(params[0])

        hs = []
        new_qs = []

        for l, q in enumerate(qs):
            h = self.posteriors[l].sample(
                q, size=(self.n_inference_samples, q.shape[0], q.shape[1]))
            hs.append(h)

        ys = [y[None, :, :]] + hs[:-1]
        p_ys = [self.p_y_given_h(h, l, *params) for l, h in enumerate(hs)]

        log_w = -self.posteriors[-1].neg_log_prob(hs[-1], prior[None, None, :])

        for l in xrange(self.n_layers):
            cond_term = -self.conditionals[l].neg_log_prob(ys[l], p_ys[l])
            post_term = -self.posteriors[l].neg_log_prob(hs[l], qs[l][None, :, :])
            log_w += cond_term - post_term

        log_w_max = T.max(log_w, axis=0, keepdims=True)
        w = T.exp(log_w - log_w_max)
        w_tilde = w / w.sum(axis=0, keepdims=True)

        for l in xrange(self.n_layers):
            h = hs[l]
            q = (w_tilde[:, :, None] * h).sum(axis=0)
            new_qs.append((1.0 - self.inference_rate) * qs[l] + self.inference_rate * q)

        cost = -T.log(w).mean()

        return tuple(new_qs) + (cost,)

    def _init_adapt(self, qs):
        return []

    def _init_variational_params_adapt(self, state):
        print 'Initializing variational params for AdIS'
        q0s = []

        for l in xrange(self.n_layers):
            state = self.posteriors[l](state)
            q0s.append(state)

        return q0s

    def _unpack_adapt(self, q0s, outs):
        if outs is not None:
            qss = outs[:-1]
            costs = outs[-1]

            if qss[0].ndim == 2:
                for i in xrange(len(qss)):
                    qs = qss[i]
                    q0 = q0s[i]
                    qs = concatenate([q0[None, :, :], qs[None, :, :]])
                    qss[i] = qs
                    costs = [costs]
            else:
                for i in xrange(len(qss)):
                    qs = qss[i]
                    q0 = q0s[i]
                    qs = concatenate([q0[None, :, :], qs])
                    qss[i] = qs
        else:
            qss = q0s
            costs = [T.constant(0.).astype(floatX)]
        return qss, costs

        qss = outs[:self.n_layers]
        qss = [concatenate([q0[None, :, :], qs]) for q0, qs in zip(q0s, qss)]
        return qss, outs[-1]

    def _params_adapt(self):
        return []

    # Momentum
    def _step_momentum(self, y, *params): pass
    def _init_momentum(self, zs): pass
    def _unpack_momentum(self, z0s, outs): pass
    def _params_momentum(self): pass
    def init_variational_params(self, state): pass

    def infer_q(self, x, y, n_inference_steps):
        updates = theano.OrderedUpdates()

        q0s = self.init_variational_params(x)
        ys = T.alloc(0., n_inference_steps + 1, y.shape[0], y.shape[1]) + y[None, :, :]
        seqs = [ys]
        outputs_info = q0s + self.init_infer(q0s) + [None]
        non_seqs = self.params_infer() + self.get_params()

        print 'Doing %d inference steps and a rate of %.2f with %d inference samples' % (n_inference_steps, self.inference_rate, self.n_inference_samples)

        if isinstance(n_inference_steps, T.TensorVariable) or n_inference_steps > 1:
            print 'Scan inference'
            outs, updates = theano.scan(
                self.step_infer,
                sequences=seqs,
                outputs_info=outputs_info,
                non_sequences=non_seqs,
                name=tools._p(self.name, 'infer'),
                n_steps=n_inference_steps,
                profile=tools.profile,
                strict=False
            )
            qss, i_costs = self.unpack_infer(q0s, outs)

        elif n_inference_steps == 1:
            print 'Simple call inference'
            inps = [ys[0]] + outputs_info[:-1] + non_seqs
            outs = self.step_infer(*inps)
            qss, i_costs = self.unpack_infer(q0s, None)
        else:
            print 'No refinement inference'
            qss, i_costs = self.unpack_infer(q0s, None)

        return (qss, i_costs), updates

    # Inference
    def inference(self, x, y, n_inference_steps=20, n_samples=100):
        (qss, _), updates = self.infer_q(x, y, n_inference_steps)

        qks = [q[-1] for q in qss]

        (prior_energy, h_energy, y_energy), m_constants = self.m_step(
            x, y, qks, n_samples=n_samples)

        constants = m_constants + qks

        return (qks, prior_energy, h_energy, y_energy), updates, constants

    def __call__(self, x, y, n_samples=100, n_inference_steps=0,
                 calculate_log_marginal=False, stride=0):

        outs = OrderedDict()
        updates = theano.OrderedUpdates()
        prior = T.nnet.sigmoid(self.z)

        (qss, i_costs), updates_i = self.infer_q(x, y, n_inference_steps)
        updates.update(updates_i)

        if n_inference_steps > stride and stride != 0:
            steps = [0] + range(n_inference_steps // stride, n_inference_steps + 1, n_inference_steps // stride)
            steps = steps[:-1] + [n_inference_steps]
        elif n_inference_steps > 0:
            steps = [0, n_inference_steps]
        else:
            steps = [0]

        lower_bounds = []
        nlls = []
        pys = []

        for i in steps:
            qs = [q[i] for q in qss]
            hs = []
            for l, q in enumerate(qs):
                h = self.posteriors[l].sample(
                    q, size=(n_samples, q.shape[0], q.shape[1]))
                hs.append(h)

            ys = [y[None, :, :]] + hs[:-1]
            p_ys = [conditional(h) for h, conditional in zip(hs, self.conditionals)]
            pys.append(p_ys[0])

            lower_bound = self.posteriors[-1].neg_log_prob(hs[-1], prior[None, None, :]).mean(axis=(0, 1))
            for l in xrange(self.n_layers):
                post_term = self.posteriors[l].entropy(qs[l])
                cond_term = self.conditionals[l].neg_log_prob(ys[l], p_ys[l]).mean(axis=0)
                lower_bound += (cond_term - post_term).mean(axis=0)

            lower_bounds.append(lower_bound)

            if calculate_log_marginal:
                log_w = -self.posteriors[-1].neg_log_prob(hs[-1], prior[None, None, :])
                for l in xrange(self.n_layers):
                    cond_term = -self.conditionals[l].neg_log_prob(ys[l], p_ys[l])
                    post_term = -self.posteriors[l].neg_log_prob(hs[l], qs[l][None, :, :])
                    log_w += cond_term - post_term

                log_w_max = T.max(log_w, axis=0, keepdims=True)
                w = T.exp(log_w - log_w_max)
                nll = -(T.log(w.mean(axis=0, keepdims=True)) + log_w_max).mean()
                nlls.append(nll)

        outs.update(
            pys=pys,
            lower_bound=lower_bounds[-1],
            lower_bounds=lower_bounds,
            lower_bound_gain=(lower_bounds[0]-lower_bounds[-1])
        )

        if calculate_log_marginal:
            outs.update(nll=nlls[-1], nlls=nlls)

        return outs, updates