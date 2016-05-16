'''
Module for Deep Helmholtz machines
'''

from collections import OrderedDict
from copy import copy
import numpy as np
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from . import Layer
from .distributions import (
    Binomial,
    Distribution,
    Gaussian,
    Laplace,
    Logistic,
    resolve as resolve_prior
)
from .mlp import MLP, resolve as resolve_mlp

from ..utils import floatX, tools
from ..utils.tools import (
    concatenate,
    init_rngs,
    init_weights,
    get_w_tilde,
    log_mean_exp,
    log_sum_exp,
    update_dict_of_lists,
    _slice
)


def unpack(dim_hs=None,
           prior=None,
           extra_args=dict(),
           data_iter=None,
           **model_args):

    '''
    Function to unpack pretrained model into DeepHelmholtz class.

    See DeepHelmholtz.factory` for details.
    '''

    distributions = data_iter.distributions
    dims = data_iter.dims
    models = []

    print 'Forming deep Helmholtz'

    model = DeepHelmholtz.factory(
        dim_hs, dims, distributions,
        prior=prior,
        rec_args=rec_args,
        gen_args=gen_args
    )

    models.append(model)
    models += model.conditionals + model.posteriors
    models.append(model.prior)

    return models, model_args, None


class DeepHelmholtz(Layer):
    def __init__(self,  posteriors, conditionals, prior,
                 name=None, **kwargs):

        self.n_layers = len(posteriors)
        assert len(posteriors) == len(conditionals)
        self.posteriors = posteriors
        self.conditionals = conditionals
        self.prior = prior
        self.dim_hs = [posterior.distribution.dim for posterior in self.posteriors]

        if name is None:
            if isinstance(prior, Binomial):
                name = 'sbn'
            elif isinstance(prior, Gaussian):
                name = 'gbn'
            elif isinstance(prior, Logistic):
                name = 'lbn'
            elif isinstance(prior, Laplace):
                name = 'labn'
            else:
                raise ValueError('Prior type %r not supported' % type(prior))

        kwargs = init_weights(self, **kwargs)
        kwargs = init_rngs(self, **kwargs)

        super(DeepHelmholtz, self).__init__(name=name)

    @staticmethod
    def factory(dim_hs, data_iter=None, distributions=None, dims=None, **kwargs):

        posteriors, conditionals, prior = DeepHelmholtz.mlp_factory(
            dim_hs, data_iter=data_iter, distributions=distributions, dims=dims,
            **kwargs)
        return DeepHelmholtz(posteriors, conditionals, prior)

    @staticmethod
    def mlp_factory(dim_hs, data_iter=None, distributions=None, dims=None,
                    prior=None, rec_args=None, gen_args=None):

        if data_iter is not None:
            distributions = data_iter.distributions
            dims = data_iter.dims
        assert dims is not None and distributions is not None

        if rec_args is None:
            rec_args = dict(input_layer=data_iter.name)
        if gen_args is None:
            gen_args = dict(output=data_iter.name)

        # Forming the prior model.
        if prior is None:
            if (rec_args is None) and (rec_args.get('distribution') is None):
                prior = 'binomial'
            else:
                prior = rec_args['distribution']
        if isinstance(prior, Distribution):
            PC = prior
        else:
            PC = resolve_prior(prior)
        prior_model = PC(dim_hs[-1])

        posteriors = []
        conditionals = []

        input_name = rec_args.get('input_layer')
        output_name = gen_args.get('output')

        for l, _ in enumerate(dim_hs):
            if l == 0:
                dim_in = dims[input_name]
            else:
                dim_in = dim_hs[l - 1]

            dim_out = dim_hs[l]

            # Forming the generation network for this layer
            gen_args['dim_in'] = dim_out
            gen_args['dim_out'] = dim_in

            t = gen_args.get('type', None)
            if t == 'darn':
                GC = DARN
            else:
                GC = resolve_mlp(t)
            if l == 0:
                gen_args['distribution'] = distributions[output_name]
            else:
                gen_args['distribution'] = rec_args['distribution']
            conditional = GC.factory(**gen_args)
            conditionals.append(conditional)

            # Forming the recogntion network for this layer
            RC = resolve_mlp(rec_args.get('type', None))
            if rec_args.get('distribution', None) is None:
                rec_args['distribution'] = prior
            rec_args['dim_in'] = dim_in
            rec_args['dim_out'] = dim_out
            posterior = RC.factory(**rec_args)
            posteriors.append(posterior)

        return posteriors, conditionals, prior_model

    # Setup --------------------------------------------------------------------
    def set_params(self):
        self.params = OrderedDict()
        for l, dim_h in enumerate(self.dim_hs):
            if l == 0:
                self.posteriors[l].name = self.name + '_posterior'
                self.conditionals[l].name = self.name + '_conditional'
            else:
                self.posteriors[l].name = self.name + '_posterior%d' % l
                self.conditionals[l].name = self.name + '_conditional%d' % l

    def set_tparams(self):
        tparams = super(DeepHelmholtz, self).set_tparams()
        tparams.update(**self.prior.set_tparams())

        for l in xrange(self.n_layers):
            tparams.update(**self.posteriors[l].set_tparams())
            tparams.update(**self.conditionals[l].set_tparams())

        return tparams

    # Fetch params -------------------------------------------------------------
    def get_params(self):
        params = self.prior.get_params()
        for l in xrange(self.n_layers):
            params += self.conditionals[l].get_params()
            params += self.posteriors[l].get_params()
        return params

    def get_prior_params(self, *params):
        params = list(params)
        return params[:self.prior.n_params]

    def get_posterior_params(self, level, *params):
        assert level < self.n_layers
        params = list(params)
        start = self.prior.n_params
        start += self.conditionals[0].n_params
        for l in xrange(level):
            start += self.posteriors[l].n_params
            start += self.conditionals[l+1].n_params
        stop = start + self.posteriors[level].n_params
        return params[start:stop]

    def get_conditional_params(self, level, *params):
        assert level < self.n_layers
        params = list(params)
        start = self.prior.n_params
        for l in xrange(level):
            start += self.conditionals[l].n_params
            start += self.posteriors[l].n_params
        stop = start + self.conditionals[level].n_params
        return params[start:stop]

    # Extra functions ----------------------------------------------------------
    def get_center(self, p):
        return self.conditionals[0].get_center(p)

    def sample_from_prior(self, n_samples=100):
        h, updates = self.prior.sample(n_samples)
        for conditional in self.conditionals[::-1]:
            p = conditional.feed(h)
            h, _ = conditional.sample(p)
            h = h[0]

        return p, updates

    def generate_from_latent(self, h, level=None):
        if level is None:
            level = self.n_layers - 1
        for l in xrange(level, -1, -1):
            p = self.conditionals[l].feed(h)
            h, _ = self.conditionals[l].sample(p)

        center = self.get_center(p)
        return center

    def visualize_latents(self):
        h0, h = self.prior.generate_latent_pair()

        for l in xrange(self.n_layers - 1, -1, -1):
            p0 = self.conditionals[l].feed(h0)
            p = self.conditionals[l].feed(h)

            h0, _ = self.conditionals[l].sample(p0)
            h, _ = self.conditionals[l].sample(p)
            h0 = h0[0]
            h = h[0]
        py = self.conditionals[0].distribution.visualize(p0, p)
        return py

    def l2_decay(self, rate):
        rec_l2_cost = T.constant(0.).astype(floatX)
        gen_l2_cost = T.constant(0.).astype(floatX)

        for l in xrange(self.n_layers):
            rec_l2_cost += self.posteriors[l].l2_decay(rate)
            gen_l2_cost += self.conditionals[l].l2_decay(rate)

        rval = OrderedDict(
            rec_l2_cost=rec_l2_cost,
            gen_l2_cost=gen_l2_cost,
            cost = rec_l2_cost + gen_l2_cost
        )

        return rval

    # --------------------------------------------------------------------------
    def p_y_given_h(self, h, level, *params):
        params = self.get_conditional_params(level, *params)
        return self.conditionals[level].step_feed(h, *params)

    def init_inference_samples(self, level, size):
        return self.posteriors[level].distribution.prototype_samples(size)

    def __call__(self, x, y, qks, n_posterior_samples=10,
                 pass_gradients=False, sample_posterior=False, reweight=False,
                 reweight_gen_only=False, sleep_phase=False):
        constants = []
        results = OrderedDict()

        # Infer from recogntion network
        q0s   = []
        state = x[None, :, :]
        for l in xrange(self.n_layers):
            q0 = self.posteriors[l].feed(state).mean(axis=0)
            q0s.append(q0)
            if sample_posterior:
                state, _ = self.posteriors[l].sample(qks[l], n_samples=n_samples)
            else:
                state = q0[None, :, :]

        if qks is None:
            qks = q0s
        elif not pass_gradients:
            constants += qks

        # Sample from posterior
        hs = []
        for l, qk in enumerate(qks):
            r = self.init_inference_samples(
                l, size=(n_posterior_samples, y.shape[0], self.dim_hs[l]))
            h = self.posteriors[l].distribution.step_sample(r, qk[None, :, :])
            hs.append(h)

        # Get the conditional distributions
        py_hs = [conditional.feed(h) for h, conditional in zip(hs, self.conditionals)]

        # Calculate posterior terms
        ys = [y[None, :, :]] + hs[:-1]
        log_py_h = T.zeros((n_posterior_samples, y.shape[0])).astype(floatX)
        log_qh0 = T.zeros_like(log_py_h)
        log_qhk = T.zeros_like(log_py_h)
        posterior_term = T.zeros_like(log_py_h)
        for l in xrange(self.n_layers):
            log_py_h -= self.conditionals[l].neg_log_prob(ys[l], py_hs[l])
            log_qh0_t = -self.posteriors[l].neg_log_prob(hs[l], q0s[l])
            log_qh0 += log_qh0_t
            log_qhk -= self.posteriors[l].neg_log_prob(hs[l], qks[l])

            if not pass_gradients:
                if self.posteriors[l].distribution.has_kl and not reweight:
                    KL_qk_q0 = self.posteriors[l].distribution.step_kl_divergence(
                        qks[l], *self.posteriors[l].distribution.split_prob(q0s[l]))
                    if results.get('KL(q_k||q_0)'):
                        results['KL(q_k||q_0)'] += KL_qk_q0
                    else:
                        results['KL(q_k||q_0)'] = KL_qk_q0
                    posterior_term += KL_qk_q0
                else:
                    if results.get('-log q(h)'):
                        results['-log q(h)'] -= log_qh0_t.mean()
                    else:
                        results['-log q(h)'] = -log_qh0_t.mean()
                    posterior_term += -log_qh0_t

        log_ph = -self.prior.neg_log_prob(hs[-1])

        prior_entropy = self.prior.entropy()
        q_entropy = T.constant(0.).astype(floatX)
        for l, qk in enumerate(qks):
            q_entropy += self.posteriors[l].entropy(qk)

        assert log_ph.ndim == log_qhk.ndim == log_py_h.ndim

        # Log marginal
        log_p = log_sum_exp(
            log_py_h + log_ph - log_qhk, axis=0) - T.log(n_posterior_samples)

        # Reconstruction term
        recon_term = -log_py_h

        # The prior term
        if self.prior.has_kl and not reweight:
            KL_qk_p = self.prior.kl_divergence(qks[-1])
            results['KL(q_k||p)'] = KL_qk_p
            KL_term = KL_qk_p
        else:
            prior_energy = -log_ph
            results['-log p(h)'] = prior_energy.mean()
            KL_term = prior_energy - self.posteriors[-1].entropy(qks[-1])

        lower_bound = -(recon_term + KL_term).mean()

        w_tilde = get_w_tilde(log_py_h + log_ph - log_qhk)
        results['log ESS'] = T.log(1. / (w_tilde ** 2).sum(0)).mean()
        if sleep_phase:
            r = self.init_inference_samples(
                (n_posterior_samples, y.shape[0], self.dim_h))
            h_s = self.prior.step_sample(
                r, self.prior.get_prob(*self.prior.get_params()))

            y_s = h_s
            for l in xrange(self.n_layers-1, -1, -1):
                py_h_s = self.conditionals[l].feed(y_s)
                y_s, _ = self.conditional.sample(py_h_s)
                y_s = y_s[0]
            constants.append(y_s)

            state = y_s
            for l in xrange(self.n_layers):
                q0_s = self.posteriors[l].feed(state).mean(axis=0)
                if sample_posterior:
                    state, _ = self.posteriors[l].sample(q0_s, n_samples=n_samples)
                else:
                    state = q0[None, :, :]
            log_qh0 = -self.posterior.neg_log_prob(h_s, q0_s)
            cost = -((w_tilde * (log_py_h + log_ph)).sum((0, 1))
                    + log_qh0.sum(1).mean(0))
            constants.append(w_tilde)
        elif reweight:
            cost = n_posterior_samples * (
                w_tilde * (recon_term - log_ph - log_qh0)).sum((0, 1))
            constants.append(w_tilde)
        elif reweight_gen_only:
            cost = -((w_tilde * (log_py_h + log_ph)).sum((0, 1))
                    + log_qh0.sum(1).mean(0))
            constants.append(w_tilde)
        else:
            cost = (recon_term + KL_term + posterior_term).sum(1).mean()

        results.update(**{
            '-log p(x|h)': recon_term.mean(),
            '-log p(x)': -log_p.mean(),
            'H(p)': prior_entropy,
            'H(q)': q_entropy.mean(0),
            'lower_bound': lower_bound,
            'cost': cost
        })

        samples = OrderedDict(
            py=py_hs[0],
            batch_energies=recon_term,
            w_tilde=w_tilde)

        return results, samples, constants, theano.OrderedUpdates()
