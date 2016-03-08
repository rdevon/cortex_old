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
from models.distributions import Binomial
from models.mlp import (
    MLP,
    MultiModalMLP
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
           prior=None,
           extra_args=dict(),
           distributions=dict(),
           dims=dict(),
           dataset_args=None,
           **model_args):

    '''
    Function to unpack pretrained model into fresh SBN class.
    '''

    print 'Unpacking model with parameters %s' % model_args.keys()
    print ('Reloading currently limited to simple posteriors and conditionals '
           '(shallow MLPs)')

    models = []
    print 'Forming deep SBN'
    model = DeepSBN(dim_in, dim_hs=dim_hs)
    models.append(model)
    models += model.conditionals + model.posteriors
    models.append(model.prior)

    return models, model_args, extra_args


class DeepSBN(Layer):
    def __init__(self, dim_in, dim_hs,
                 posteriors=None, conditionals=None,
                 prior=None,
                 name='sbn',
                 **kwargs):

        self.dim_in = dim_in
        self.dim_hs = dim_hs
        self.n_layers = len(self.dim_hs)
        self.posteriors = posteriors
        self.conditionals = conditionals
        self.prior = prior

        kwargs = init_weights(self, **kwargs)
        kwargs = init_rngs(self, **kwargs)

        super(DeepSBN, self).__init__(name=name)

    def set_params(self):
        self.params = OrderedDict()

        if self.prior is None:
            self.prior = Binomial(self.dim_hs[-1])

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
                    dim_in, dim_h,
                    rng=self.rng, trng=self.trng,
                    distribution='binomial')

            if self.conditionals[l] is None:
                self.conditionals[l] = MLP(
                    dim_h, dim_in,
                    rng=self.rng, trng=self.trng,
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
        return self.conditionals[level].step_feed(h, *params)

    def sample_from_prior(self, n_samples=100):
        h, updates = self.prior.sample(n_samples)
        for conditional in self.conditionals[::-1]:
            p = conditional.feed(h)
            h, _ = conditional.sample(p)
            h = h[0]

        return p, updates

    def l2_decay(self, rate):
        rec_l2_cost = T.constant(0.).astype(floatX)
        gen_l2_cost = T.constant(0.).astype(floatX)

        for l in xrange(self.n_layers):
            rec_l2_cost += self.posteriors[l].get_L2_weight_cost(rate)
            gen_l2_cost += self.conditionals[l].get_L2_weight_cost(rate)

        rval = OrderedDict(
            rec_l2_cost=rec_l2_cost,
            gen_l2_cost=gen_l2_cost,
            cost = rec_l2_cost + gen_l2_cost
        )

        return rval

    def __call__(self, x, y, qks, n_posterior_samples=10, sample_posterior=False):
        constants = qks

        q0s   = []
        state = x[None, :, :]
        for l in xrange(self.n_layers):
            q0 = self.posteriors[l].feed(state).mean(axis=0)
            q0s.append(q0)
            if sample_posterior:
                raise NotImplementedError()
                state, _ = self.posteriors[l].sample(qks[l], n_samples=n_samples)
            else:
                state = q0[None, :, :]

        hs = []
        for l, qk in enumerate(qks):
            r = self.trng.uniform(
                (n_posterior_samples, y.shape[0], self.dim_hs[l]), dtype=floatX)
            h = (r <= qk[None, :, :]).astype(floatX)
            hs.append(h)

        h0s = []
        for l, q0 in enumerate(q0s):
            r = self.trng.uniform((n_posterior_samples, y.shape[0], self.dim_hs[l]), dtype=floatX)
            h = (r <= q0[None, :, :]).astype(floatX)
            h0s.append(h)

        p_ys = [conditional.feed(h) for h, conditional in zip(hs, self.conditionals)]
        p_y0s = [conditional.feed(h0) for h0, conditional in zip(h0s, self.conditionals)]
        ys   = [y[None, :, :]] + hs[:-1]
        y0s   = [y[None, :, :]] + h0s[:-1]

        log_py_h = T.constant(0.).astype(floatX)
        log_py_h0 = T.constant(0.).astype(floatX)
        log_qh   = T.constant(0.).astype(floatX)
        log_qh0  = T.constant(0.).astype(floatX)
        log_qkh  = T.constant(0.).astype(floatX)
        for l in xrange(self.n_layers):
            log_py_h -= self.conditionals[l].neg_log_prob(ys[l], p_ys[l])
            log_py_h0 -= self.conditionals[l].neg_log_prob(y0s[l], p_y0s[l])
            log_qh   -= self.posteriors[l].neg_log_prob(hs[l], q0s[l])
            log_qh0  -= self.posteriors[l].neg_log_prob(h0s[l], q0s[l])
            log_qkh  -= self.posteriors[l].neg_log_prob(hs[l], qks[l])
        log_ph = -self.prior.neg_log_prob(hs[-1])
        log_ph0 = -self.prior.neg_log_prob(h0s[-1])

        assert log_ph.ndim == log_qh.ndim == log_py_h.ndim

        log_p         = log_sum_exp(log_py_h + log_ph - log_qkh, axis=0) - T.log(n_posterior_samples)
        log_p0        = log_sum_exp(log_py_h0 + log_ph0 - log_qh0, axis=0) - T.log(n_posterior_samples)

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
            'log p0': log_p0.mean(0),
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

        return results, samples, theano.OrderedUpdates()
