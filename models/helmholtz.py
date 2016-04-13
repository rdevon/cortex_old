'''
Module for Helmhotlz machines.

SBNs, GBNs, etc. Networks with DARN components should also be trainable
with this class.
'''

from collections import OrderedDict
import numpy as np
import theano
from theano import tensor as T

from . import Layer
from darn import AutoRegressor, DARN
from distributions import (
    Binomial,
    Gaussian,
    Logistic,
    resolve as resolve_prior
)
from mlp import MLP, MultiModalMLP, resolve as resolve_mlp
from utils import floatX, tools
from utils.tools import (
    concatenate,
    init_rngs,
    init_weights,
    log_mean_exp,
    log_sum_exp,
    update_dict_of_lists,
    _slice
)


def unpack(dim_h=None,
           prior=None,
           rec_args=None,
           gen_args=None,
           data_iter=None,
           **model_args):
    '''
    Function to unpack pretrained model into fresh Helmholtz class.

    See `Helmholtz.factory` for details.
    '''

    distributions = data_iter.distributions
    dims = data_iter.dims
    models = []

    print 'Forming Helmholtz machine'
    model = Helmholtz.factory(
        dim_h, dims, distributions,
        prior=prior,
        rec_args=rec_args,
        gen_args=gen_args)

    models.append(model)
    models += [model.posterior, model.conditional, model.prior]
    return models, model_args, None


class Helmholtz(Layer):
    '''Generic Helmholtz machine class.

    SBNs, GBNs (VAE), and other prior distributions supported here.
    Only single latent layer for this class. See `DeepHelmholtz`.
    Use Helmholtz.factory for simple SBN.

    Attributes:
        posterior: MLP, approximate posterior distribution network.
            (aka inference net or recognition net)
        conditional: MLP, conditional network.
        prior: Distribution, prior distribution of latent variables.
    '''
    def __init__(self, posterior, conditional, prior,
                 name=None, **kwargs):
        '''Init function for MLPs.

        Args:
            posterior: MLP, approximate posterior distribution network.
                (aka inference net or recognition net)
            conditional: MLP, conditional network.
            prior: Distribution, prior distribution of latent variables.
        '''

        self.posterior = posterior
        self.conditional = conditional
        self.prior = prior
        self.dim_h = self.prior.dim

        if name is None:
            if isinstance(prior, Binomial):
                name = 'sbn'
            elif isinstance(prior, Gaussian):
                name = 'gbn'
            elif isinstance(prior, Logistic):
                name = 'lbn'
            else:
                raise ValueError('Prior type %r not supported' % type(prior))

        kwargs = init_weights(self, **kwargs)
        kwargs = init_rngs(self, **kwargs)

        super(Helmholtz, self).__init__(name=name)

    @staticmethod
    def factory(dim_h, data_iter, **kwargs):
        '''Factory for forming conditional, posterior, and prior.

        Args:
            dim_h: int, latent dimension.
            data_iter: Dataset class.
            **kwargs: kwargs for mlp_factory
        Returns:
            Helmholtz object.
        '''

        posterior, conditional, prior = Helmholtz.mlp_factory(
            dim_h, data_iter, **kwargs)
        return Helmholtz(posterior, conditional, prior)

    @staticmethod
    def mlp_factory(dim_h, data_iter, prior=None, rec_args=None, gen_args=None):
        '''Factory for forming conditional, posterior, and prior.

        Args:
            dim_h: int, latent dimension.
            data_iter: Dataset class.
            prior: str (optional), type of prior. See `distributions.py`.
            rec_args: dict (optional), arguments for approximate posterior.
            gen_args: dict (optional), arguments for conditional network.
        Returns:
            conditional: MLP.
            posterior: MLP.
            prior: Distribution.
        '''

        distributions = data_iter.distributions
        dims = data_iter.dims

        if rec_args is None:
            rec_args = dict(input_layer=distributions[data_iter.name])
        if gen_args is None:
            gen_args = dict(output=distributions[data_iter.name])

        # Forming the prior model.
        if prior is None:
            if (rec_args is None) or (rec_args.get('distribution') is None):
                prior = 'binomial'
            else:
                prior = rec_args['distribution']
        PC = resolve_prior(prior)
        prior_model = PC(dim_h)

        # Forming the recogntion network, aka posterior
        RC = resolve_mlp(rec_args.get('type', None))
        input_name = rec_args.get('input_layer')
        if rec_args.get('distribution', None) is None:
            rec_args['distribution'] = prior
        rec_args['dim_in'] = dims[input_name]
        rec_args['dim_out'] = dim_h
        posterior = RC.factory(**rec_args)

        # Forming the generative network, aka conditional
        output_name = gen_args['output']
        gen_args['dim_in'] = dim_h

        t = gen_args.get('type', None)
        if t == 'darn':
            GC = DARN
        else:
            GC = resolve_mlp(t)
        if t == 'dag':
            for out in gen_args['graph']['outputs']:
                gen_args['graph']['outs'][out] = dict(
                    dim=dims[out],
                    distribution=distributions[out])
        else:
            gen_args['dim_out'] = dims[output_name]
            gen_args['distribution'] = distributions[output_name]
        conditional = GC.factory(**gen_args)

        return posterior, conditional, prior_model

    def set_params(self):
        self.params = OrderedDict()
        self.prior.name = self.name + '_' + self.prior.name
        self.posterior.name = self.name + '_posterior'
        self.conditional.name = self.name + '_conditional'

    def set_tparams(self):
        tparams = super(Helmholtz, self).set_tparams()
        tparams.update(**self.posterior.set_tparams())
        tparams.update(**self.conditional.set_tparams())
        tparams.update(**self.prior.set_tparams())

        return tparams

    # Fetch params -------------------------------------------------------------
    def get_params(self):
        '''Get model parameters.'''
        params = (self.prior.get_params()
                  + self.conditional.get_params()
                  + self.posterior.get_params())
        return params

    def get_prior_params(self, *params):
        '''Get the prior params for scan.'''
        params = list(params)
        return params[:self.prior.n_params]

    def get_posterior_params(self, *params):
        '''Get the posterior params for scan.'''
        params = list(params)
        start = self.prior.n_params + self.conditional.n_params
        stop = start + self.posterior.n_params
        return params[start:stop]

    # E ---------------------------------------------------------
    def sample_from_prior(self, n_samples=100):
        '''Samples from the prior distribution.'''
        h, updates = self.prior.sample(n_samples)
        return self.conditional.feed(h), updates

    def generate_from_latent(self, h):
        '''Generates an image from a latent state.'''
        py = self.conditional.feed(h)
        center = self.conditional.get_center(py)
        return center

    def visualize_latents(self):
        '''Visualizes influence of latent variables on input space.

        Takes the difference between "on" and "off" units (defined by prior).
        See `distributions.py` for details.
        '''
        h0, h = self.prior.generate_latent_pair()
        p0 = self.conditional.feed(h0)
        p = self.conditional.feed(h)
        py = self.conditional.distribution.visualize(p0, p)
        return py

    # Misc --------------------------------------------------------------------
    def get_center(self, p):
        '''Returns the center of the conditional distribution.'''
        return self.conditional.get_center(p)

    def log_marginal(self, y, h, py, q):
        '''Computes the approximate log marginal.

        Uses \log \sum p / q - \log N

        Args:
            y: T.tensor, target values.
            h: T.tensor, latent samples.
            py: T.tesnor, conditional density p(y | h)
            q: approximate posterior q(h | y)
        Returns:
            approximate log marginal.
        '''
        log_py_h = -self.conditional.neg_log_prob(y, py)
        log_ph   = -self.prior.neg_log_prob(h)
        log_qh   = -self.posterior.neg_log_prob(h, q)
        assert log_py_h.ndim == log_ph.ndim == log_qh.ndim

        log_p     = log_py_h + log_ph - log_qh
        log_p_max = T.max(log_p, axis=0, keepdims=True)
        w         = T.exp(log_p - log_p_max)

        return (T.log(w.mean(axis=0, keepdims=True)) + log_p_max).mean()

    def l2_decay(self, rate):
        '''Get L2 decay costs.'''
        rec_l2_cost = self.posterior.get_L2_weight_cost(rate)
        gen_l2_cost = self.conditional.get_L2_weight_cost(rate)

        rval = OrderedDict(
            rec_l2_cost=rec_l2_cost,
            gen_l2_cost=gen_l2_cost,
            cost = rec_l2_cost + gen_l2_cost
        )

        return rval

    # --------------------------------------------------------------------
    def p_y_given_h(self, h, *params):
        '''p(y | h) for scan.'''
        start  = self.prior.n_params
        stop   = start + self.conditional.n_params
        params = params[start:stop]
        return self.conditional.step_feed(h, *params)

    def init_inference_samples(self, size):
        '''Initializes the samples for inference.'''
        return self.posterior.distribution.prototype_samples(size)

    def __call__(self, x, y, qk=None, n_posterior_samples=10,
                 pass_gradients=False):
        '''Call function.

        Calculates the lower bound, log marginal, and other useful quantities.
        If this is TMI for your needs, just omit what you don't need from the
        final graph.

        Args:
            x: T.tensor, input to recogntion network.
            y: T.tensor, output from conditional.
            qk: T.tensor (optional), approximate posterior parameters.
                If None, calculate from recognition network.
            n_posterior_samples: int, number of samples to use for lower bound
                and log marginal estimates.
            pass_gradients: bool, for priors with continuous distributions,
                this can facilitate learning. Otherwise, q_k should be provided.
        Returns:
            results: OrderedDict, float results.
            samples: OrderedDict, array results
                (such as samples from conditional).
            updates: OrderedUpdates.
            constants: list, for omitting quantities from passing gradients.
        '''
        constants = []
        results = OrderedDict()

        q0  = self.posterior.feed(x)
        if qk is None:
            qk = q0

        r = self.init_inference_samples(
            (n_posterior_samples, y.shape[0], self.dim_h))
        h = self.posterior.distribution.step_sample(r, qk[None, :, :])
        py_h = self.conditional.feed(h)

        log_py_h = -self.conditional.neg_log_prob(y[None, :, :], py_h)
        log_ph = -self.prior.neg_log_prob(h)
        log_qh0 = -self.posterior.neg_log_prob(h, q0[None, :, :])
        log_qkh = -self.posterior.neg_log_prob(h, qk[None, :, :])
        prior_entropy = self.prior.entropy()
        q_entropy = self.posterior.entropy(qk)

        # Log marginal
        log_p = log_sum_exp(
            log_py_h + log_ph - log_qkh, axis=0) - T.log(n_posterior_samples)

        y_energy = -log_py_h.mean(axis=0)
        cost = y_energy
        lower_bound = -y_energy

        # Some prior distributions have a tractable KL divergence.
        if self.prior.has_kl:
            KL_qk_p = self.prior.kl_divergence(qk)
            cost += KL_qk_p
            results['KL(q_k||p)'] = KL_qk_p.mean(0)
            lower_bound -= KL_qk_p
        else:
            prior_energy = -log_ph.mean(axis=0)
            cost += prior_energy
            results['-log p(h)'] = prior_energy.mean(0)
            lower_bound -= (prior_energy - q_entropy)

        # If we pass the gradients we don't want to include the KL(q_k||q_0)
        if not pass_gradients:
            if self.posterior.distribution.has_kl:
                qk_c = qk.copy()
                KL_qk_q0 = self.posterior.distribution.step_kl_divergence(
                    qk_c, *self.posterior.distribution.split_prob(q0))
                cost += KL_qk_q0
                results['KL(q_k||q_0)'] = KL_qk_q0.mean(0)
            else:
                h_energy = -log_qhk.mean(axis=0)
                h_energy_mean = h_energy.mean(axis=0)
                cost += h_energy
                results['-log q(h)'] = h_energy.mean(0),

        results.update(**{
            '-log p(x|h)': y_energy.mean(0),
            '-log p(x)': -log_p.mean(0),
            'H(p)': prior_entropy,
            'H(q)': q_entropy.mean(0),
            'lower_bound': lower_bound.mean(),
            'cost': cost.sum(0)
        })

        samples = OrderedDict(
            py=py_h,
            batch_energies=y_energy
        )

        return results, samples, theano.OrderedUpdates(), constants
