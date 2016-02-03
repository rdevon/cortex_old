'''
Module for Deep Sigmoid Belief Networks
'''

from collections import OrderedDict
from copy import copy
import numpy as np
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from layers import Layer
from models.distributions import Bernoulli
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
    update_dict_of_lists,
    _slice
)


def unpack(dim_hs=None,
           dim_in=None,
           z_init=None,
           recognition_net=None,
           generation_net=None,
           n_inference_steps=None,
           inference_method=None,
           inference_rate=None,
           extra_inference_args=dict(),
           n_inference_samples=None,
           input_mode=None, prior=None, n_layers=None, dataset=None, dataset_args=None,
           **model_args):

    dim_in = int(dim_in)
    dataset_args = dataset_args[()]

    kwargs = dict(
        inference_method=inference_method,
        inference_rate=inference_rate,
        n_inference_steps=n_inference_steps,
        z_init=z_init,
        n_inference_samples=n_inference_samples,
        extra_inference_args=extra_inference_args
    )

    out_act = 'T.nnet.sigmoid'

    models = []
    model = DeepSBN(dim_in, dim_hs=dim_hs,
            conditionals=None,
            posteriors=None,
            **kwargs)
    models.append(model)
    models += model.conditionals
    models += model.posteriors
    models.append(model.prior)

    return models, model_args, dict(dataset=dataset, dataset_args=dataset_args)


class DeepSBN(Layer):
    def __init__(self, dim_in, dim_h=None, n_layers=2, dim_hs=None,
                 posteriors=None, conditionals=None,
                 prior=None,
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
        self.prior = prior

        self.z_init = z_init

        kwargs = init_inference_args(self, **kwargs)
        kwargs = init_weights(self, **kwargs)
        kwargs = init_rngs(self, **kwargs)

        super(DeepSBN, self).__init__(name=name)

    def set_params(self):
        self.params = OrderedDict()

        if self.prior is None:
            self.prior = Bernoulli(self.dim_hs[-1])

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
                    dim_in, dim_h, dim_h=dim_h, n_layers=1,
                    rng=self.rng, trng=self.trng,
                    h_act='T.nnet.softplus',
                    out_act='T.nnet.sigmoid')

            if self.conditionals[l] is None:
                self.conditionals[l] = MLP(
                    dim_h, dim_in, dim_h=dim_h, n_layers=1,
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
        tparams.update(**self.prior.set_tparams())

        for l in xrange(self.n_layers):
            tparams.update(**self.posteriors[l].set_tparams())
            tparams.update(**self.conditionals[l].set_tparams())

        tparams = OrderedDict((k, v) for k, v in tparams.iteritems()
            if k not in excludes)

        return tparams

    def get_params(self):
        params = self.prior.get_params()
        for l in xrange(self.n_layers):
            params += self.conditionals[l].get_params()
        return params

    def get_prior_params(self, *params):
        params = list(params)
        return params[:self.prior.n_params]

    def p_y_given_h(self, h, level, *params):
        start = 1
        for l in xrange(level):
            start += self.conditionals[l].n_params
        end = start + self.conditionals[level].n_params

        params = params[start:end]
        return self.conditionals[level].step_call(h, *params)

    def sample_from_prior(self, n_samples=100):
        h, updates = self.prior.sample(n_samples)
        for conditional in self.conditionals[::-1]:
            p = conditional(h)
            h, _ = conditional.sample(p)
            h = h[0]

        return p, updates

    def step_infer(self, *params):
        raise NotImplementedError()
    def init_infer(self, z):
        raise NotImplementedError()
    def unpack_infer(self, outs):
        raise NotImplementedError()
    def params_infer(self):
        raise NotImplementedError()

    # Importance Sampling
    def _step_adapt(self, *params):
        print 'AdIS'
        params       = list(params)
        rs           = params[:self.n_layers]
        qs           = params[self.n_layers:2*self.n_layers]
        y            = params[2*self.n_layers]
        params       = params[1+2*self.n_layers:]
        prior_params = self.get_prior_params(*params)

        hs     = []
        new_qs = []

        for l, (q, r) in enumerate(zip(qs, rs)):
            h = (r <= q[None, :, :]).astype(floatX)
            hs.append(h)

        ys   = [y[None, :, :]] + hs[:-1]
        p_ys = [self.p_y_given_h(h, l, *params) for l, h in enumerate(hs)]

        log_ph = -self.prior.step_neg_log_prob(hs[-1], *prior_params)
        log_py_h = T.constant(0.).astype(floatX)
        log_qh = T.constant(0.).astype(floatX)
        for l in xrange(self.n_layers):
            log_py_h += -self.conditionals[l].neg_log_prob(ys[l], p_ys[l])
            log_qh += -self.posteriors[l].neg_log_prob(hs[l], qs[l][None, :, :])

        log_p     = log_py_h + log_ph - log_qh
        log_p_max = T.max(log_p, axis=0, keepdims=True)

        w       = T.exp(log_p - log_p_max)
        w_tilde = w / w.sum(axis=0, keepdims=True)
        cost = w.mean()

        for q, h in zip(qs, hs):
            q_ = (w_tilde[:, :, None] * h).sum(axis=0)
            new_qs.append(self.inference_rate * q_ + (1 - self.inference_rate) * q)

        return tuple(new_qs) + (cost,)

    def _init_adapt(self, qs):
        return []

    def init_q(self, state, sample_posterior=False, n_samples=10):
        print 'Initializing variational params for AdIS'
        q0s = []
        state = state[None, :, :]

        for l in xrange(self.n_layers):
            q0 = self.posteriors[l](state).mean(axis=0)
            q0s.append(q0)
            if sample_posterior:
                state, _ = self.posteriors[l].sample(q0, n_samples=n_samples)
            else:
                state = q0[None, :, :]

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

    # Learning -----------------------------------------------------------------

    def infer_q(self, x, y, n_inference_steps, n_inference_samples,
                sample_posterior=False):
        updates = theano.OrderedUpdates()

        q0s          = self.init_q(x,
                                   sample_posterior=sample_posterior,
                                   n_samples=n_inference_samples)
        constants    = q0s
        outputs_info = q0s + [None]
        non_seqs     = [y] + self.params_infer() + self.get_params()

        print 'Doing %d inference steps and a rate of %.2f with %d inference samples' % (n_inference_steps, self.inference_rate, n_inference_samples)

        if isinstance(n_inference_steps, T.TensorVariable) or n_inference_steps > 1:
            rs  = []
            for l in xrange(self.n_layers):
                r = self.trng.uniform((n_inference_steps,
                                       n_inference_samples,
                                       y.shape[0],
                                       self.dim_hs[l]), dtype=floatX)
                rs.append(r)
            seqs = rs
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
            for l in xrange(self.n_layers):
                r = self.trng.uniform((n_inference_samples,
                                       y.shape[0],
                                       self.dim_hs[l]), dtype=floatX)
                rs.append(r)
            print 'Simple call inference'
            inps = rs + outputs_info[:-1] + non_seqs
            outs = self.step_infer(*inps)
            qss, i_costs = self.unpack_infer(q0s, None)
        else:
            print 'No refinement inference'
            qss, _ = self.unpack_infer(q0s, None)
            icosts = [T.constant(0.).astype(floatX)]

        return (qss, q0s[0]), constants, updates

    def m_step(self, x, y, qks, n_samples=10, sample_posterior=False):
        constants = qks

        q0s   = []
        state = x[None, :, :]
        for l in xrange(self.n_layers):
            q0 = self.posteriors[l](state).mean(axis=0)
            q0s.append(q0)
            if sample_posterior:
                state, _ = self.posteriors[l].sample(qks[l], n_samples=n_samples)
            else:
                state = q0[None, :, :]

        hs = []
        for l, qk in enumerate(qks):
            r = self.trng.uniform((n_samples, y.shape[0], self.dim_hs[l]), dtype=floatX)
            h = (r <= qk[None, :, :]).astype(floatX)
            hs.append(h)

        p_ys = [conditional(h) for h, conditional in zip(hs, self.conditionals)]
        ys   = [y[None, :, :]] + hs[:-1]

        log_py_h = T.constant(0.).astype(floatX)
        log_qh   = T.constant(0.).astype(floatX)
        log_qkh  = T.constant(0.).astype(floatX)
        for l in xrange(self.n_layers):
            log_py_h -= self.conditionals[l].neg_log_prob(ys[l], p_ys[l])
            log_qh   -= self.posteriors[l].neg_log_prob(hs[l], q0s[l])
            log_qkh  -= self.posteriors[l].neg_log_prob(hs[l], qks[l])
        log_ph = -self.prior.neg_log_prob(hs[-1])

        assert log_ph.ndim == log_qh.ndim == log_py_h.ndim

        log_p         = log_sum_exp(log_py_h + log_ph - log_qkh, axis=0) - T.log(n_samples)

        y_energy      = -log_py_h.mean(axis=0)
        prior_energy  = -log_ph.mean(axis=0)
        h_energy      = -log_qh.mean(axis=0)

        nll           = -log_p
        prior_entropy = self.prior.entropy()
        q_entropy     = T.constant(0.).astype(floatX)
        for l, qk in enumerate(qks):
            q_entropy += self.posteriors[l].entropy(qk)

        assert prior_energy.ndim == h_energy.ndim == y_energy.ndim, (prior_energy.ndim, h_energy.ndim, y_energy.ndim)

        cost = (y_energy + prior_energy + h_energy).sum(0)
        lower_bound = (y_energy + prior_energy - q_entropy).mean()

        results = OrderedDict({
            '-log p(x|h)': y_energy.mean(0),
            '-log p(h)': prior_energy.mean(0),
            '-log q(h)': h_energy.mean(0),
            '-log p(x)': nll.mean(0),
            'H(p)': prior_entropy,
            'H(q)': q_entropy.mean(0),
            'lower_bound': lower_bound,
            'cost': cost
        })

        samples = OrderedDict(
            py=p_ys[0]
        )

        constants = qks
        return results, samples, theano.OrderedUpdates(), constants

    def rws(self, x, y, n_samples=10, qks=None, sample_posterior=False):
        print 'Doing RWS, %d samples' % n_samples
        qs   = []
        qcs   = []
        state = x[None, :, :]
        for l in xrange(self.n_layers):
            q = self.posteriors[l](state).mean(axis=0)
            qs.append(q)
            if sample_posterior:
                state, _ = self.posteriors[l].sample(q, n_samples=n_samples)
            else:
                state = q[None, :, :]
            if qks is None:
                qcs.append(q.copy())
            else:
                qcs.append(qks[l])

        hs = []
        for l, qc in enumerate(qcs):
            r = self.trng.uniform((n_samples, y.shape[0], self.dim_hs[l]), dtype=floatX)
            h = (r <= qc[None, :, :]).astype(floatX)
            hs.append(h)

        p_ys = [conditional(h) for h, conditional in zip(hs, self.conditionals)]
        ys   = [y[None, :, :]] + hs[:-1]

        log_py_h = T.constant(0.).astype(floatX)
        log_qh   = T.constant(0.).astype(floatX)
        log_qch  = T.constant(0.).astype(floatX)
        for l in xrange(self.n_layers):
            log_py_h -= self.conditionals[l].neg_log_prob(ys[l], p_ys[l])
            log_qh -= self.posteriors[l].neg_log_prob(hs[l], qs[l])
            log_qch -= self.posteriors[l].neg_log_prob(hs[l], qcs[l])
        log_ph = -self.prior.neg_log_prob(hs[-1])

        assert log_py_h.ndim == log_ph.ndim == log_qh.ndim

        log_p = log_sum_exp(log_py_h + log_ph - log_qch, axis=0) - T.log(n_samples)

        log_pq   = log_py_h + log_ph - log_qh - T.log(n_samples)
        w_norm   = log_sum_exp(log_pq, axis=0)
        log_w    = log_pq - T.shape_padleft(w_norm)
        w_tilde  = T.exp(log_w)

        y_energy      = -(w_tilde * log_py_h).sum(axis=0)
        prior_energy  = -(w_tilde * log_ph).sum(axis=0)
        h_energy      = -(w_tilde * log_qh).sum(axis=0)

        nll           = -log_p
        prior_entropy = self.prior.entropy()
        q_entropy     = T.constant(0.).astype(floatX)
        for l, qc in enumerate(qcs):
            q_entropy += self.posteriors[l].entropy(qc)

        cost = (y_energy + prior_energy + h_energy).sum(0)
        lower_bound = (y_energy + prior_energy - q_entropy).mean()

        results = OrderedDict({
            '-log p(x|h)': y_energy.mean(0),
            '-log p(h)': prior_energy.mean(0),
            '-log q(h)': h_energy.mean(0),
            '-log p(x)': nll.mean(0),
            'H(p)': prior_entropy,
            'H(q)': q_entropy.mean(0),
            'lower_bound': lower_bound,
            'cost': cost
        })

        samples = OrderedDict(
            py=p_ys[0]
        )

        constants = [w_tilde] + qcs
        return results, samples, theano.OrderedUpdates(), constants

    # Inference
    def inference(self, x, y, n_inference_steps=20, n_inference_samples=20,
                  n_samples=100, sample_posterior=False):
        if sample_posterior:
            print 'Sampling posterior for inference (SBN) at each level'
        constants = []
        if self.inference_method == 'rws' and n_inference_steps == 0:
            print 'RWS'
            results, _, updates, m_constants = self.rws(
                x, y, n_samples=n_samples, sample_posterior=sample_posterior)
            constants += m_constants
        elif self.inference_method == 'rws':
            print 'AIR and RWS'
            (qss, _), q_constants, updates = self.infer_q(
                x, y, n_inference_steps, n_inference_samples=n_inference_samples,
                sample_posterior=sample_posterior)
            qks = [q[-1] for q in qss]
            results, _, updates, m_constants = self.rws(
                x, y, n_samples=n_samples, qks=qks,
                sample_posterior=sample_posterior)
            constants = qss + m_constants + q_constants
        elif self.inference_method == 'adaptive':
            print 'AIR'
            (qss, i_costs), q_constants, updates = self.infer_q(
                x, y, n_inference_steps,
                n_inference_samples=n_inference_samples,
                sample_posterior=sample_posterior)
            qks = [q[-1] for q in qss]
            results, _, updates_m, m_constants = self.m_step(
                x, y, qks, n_samples=n_samples,
                sample_posterior=sample_posterior)
            updates.update(updates_m)
            constants = qss + m_constants + q_constants

        return results, updates, constants

    def __call__(self, x, y, n_samples=100, n_inference_steps=0,
                 n_inference_samples=20, stride=10,
                 sample_posterior=False):

        (qss, i_costs), _, updates = self.infer_q(
            x, y, n_inference_steps, n_inference_samples=n_inference_samples,
            sample_posterior=sample_posterior)

        if n_inference_steps > stride and stride != 0:
            steps = [0] + range(stride, n_inference_steps, stride)
            steps = steps[:-1] + [n_inference_steps - 1]
        elif n_inference_steps > 0:
            steps = [0, n_inference_steps - 1]
        else:
            steps = [0]

        full_results = OrderedDict()
        samples = OrderedDict()
        for i in steps:
            qks  = [qks[i] for qks in qss]
            results, samples_k, updates_m, _ = self.m_step(
                x, y, qks, n_samples=n_samples,
                sample_posterior=sample_posterior)
            updates.update(updates_m)
            update_dict_of_lists(full_results, **results)
            update_dict_of_lists(samples, **samples_k)

        results = OrderedDict()
        for k, v in full_results.iteritems():
            results[k] = v[-1]
            results[k + '0'] = v[0]
            results['d_' + k] = v[0] - v[-1]

        return results, samples, full_results, updates


class DeepSBN_AR(DeepSBN):
    def __init__(self, dim_in,
                 **kwargs):

        super(DeepSBN_AR, self).__init__(dim_in, **kwargs)

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
                dim_in_post = dim_in
            else:
                dim_in = self.dim_hs[l-1]
                dim_in_post += dim_in

            if self.posteriors[l] is None:
                self.posteriors[l] = MLP(
                    dim_in_post, dim_h, dim_h, 1,
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

    def m_step(self, x, y, qks, n_samples=10):
        constants = []

        hs = []
        for l, qk in enumerate(qks):
            h, _ = self.posteriors[l].sample(qk, size=(n_samples, qk.shape[0], qk.shape[1]))
            hs.append(h)

        p_ys = [conditional(h) for h, conditional in zip(hs, self.conditionals)]
        ys   = [y[None, :, :]] + hs[:-1]
        q0s  = self.init_variational_params(x)

        prior_energy       = self.prior.neg_log_prob(qks[-1])
        conditional_energy = T.constant(0.).astype(floatX)
        posterior_energy   = T.constant(0.).astype(floatX)
        for l in xrange(self.n_layers):
            posterior_energy += self.posteriors[l].neg_log_prob(qks[l], q0s[l])
            conditional_energy += self.conditionals[l].neg_log_prob(
                ys[l], p_ys[l]).mean(axis=0)

        return (prior_energy.mean(axis=0), posterior_energy.mean(axis=0),
                conditional_energy.mean(axis=0)), constants

    def _init_variational_params_adapt(self, state):
        print 'Initializing variational params for AdIS'
        q0s = []

        ndim = state.ndim
        state = [state]
        for l in xrange(self.n_layers):
            y = self.posteriors[l](concatenate(state, axis=ndim-1))
            q0s.append(y)
            state.append(y)

        return q0s
