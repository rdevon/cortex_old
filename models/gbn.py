'''
Module of Gaussian Belief Networks
'''

from collections import OrderedDict
import numpy as np
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from distributions import Gaussian
from layers import Layer
from mlp import MLP
from utils import tools
from utils.tools import (
    concatenate,
    floatX,
    init_rngs,
    init_weights,
    log_mean_exp,
    log_sum_exp,
    _slice
)


def unpack(dim_in=None,
           dim_h=None,
           recognition_net=None,
           generation_net=None,
           extra_args=dict(),
           distributions=dict(),
           dims=dict(),
           dataset_args=dict(),
           center_input=None,
           **model_args):

    print 'Unpacking model with parameters %s' % model_args.keys()

    print 'Forming prior model'
    prior_model = Gaussian(dim_h)
    models = []

    print 'Forming MLPs'
    kwargs = GBN.mlp_factory(dim_h, dims, distributions,
                           recognition_net=recognition_net,
                           generation_net=generation_net)

    models += kwargs.values()
    kwargs['prior'] = prior_model
    models.append(prior_model)
    print 'Forming GBN'
    model = GBN(dim_in, dim_h, **kwargs)
    models.append(model)
    return models, model_args, extra_args


class GBN(Layer):
    def __init__(self, dim_in, dim_h,
                 posterior=None, conditional=None, prior=None,
                 name='gbn',
                 **kwargs):

        self.dim_in = dim_in
        self.dim_h = dim_h

        self.posterior = posterior
        self.conditional = conditional
        self.prior = prior

        kwargs = init_weights(self, **kwargs)
        kwargs = init_rngs(self, **kwargs)

        super(GBN, self).__init__(name=name)

    @staticmethod
    def mlp_factory(dim_h, dims, distributions,
                    recognition_net=None, generation_net=None):
        mlps = {}

        if recognition_net is not None:
            print recognition_net
            input_name = recognition_net.get('input_layer')
            recognition_net['distribution'] = 'gaussian'
            recognition_net['dim_in'] = dims[input_name]
            recognition_net['dim_out'] = dim_h
            posterior = MLP.factory(**recognition_net)
            mlps['posterior'] = posterior

        if generation_net is not None:
            print generation_net
            output_name = generation_net['output']
            generation_net['dim_in'] = dim_h

            t = generation_net.get('type', None)
            if t is None:
                generation_net['dim_out'] = dims[output_name]
                generation_net['distribution'] = distributions[output_name]
                conditional = MLP.factory(**generation_net)
            elif t == 'MMMLP':
                for out in generation_net['graph']['outputs']:
                    generation_net['graph']['outs'][out] = dict(
                        dim=dims[out],
                        distribution=distributions[out])
                conditional = MultiModalMLP.factory(**generation_net)
            else:
                raise ValueError(t)
            mlps['conditional'] = conditional

        return mlps

    def set_params(self):
        self.params = OrderedDict()

        if self.prior is None:
            self.prior = Gaussian(self.dim_h)

        if self.posterior is None:
            self.posterior = MLP(self.dim_in, self.dim_h,
                                 dim_hs=[],
                                 rng=self.rng, trng=self.trng,
                                 h_act='T.nnet.sigmoid',
                                 distribution='gaussian')
        if self.conditional is None:
            self.conditional = MLP(self.dim_h, self.dim_in,
                                   dim_hs=[],
                                   rng=self.rng, trng=self.trng,
                                   h_act='T.nnet.sigmoid',
                                   distribution='binomial')

        self.posterior.name = self.name + '_posterior'
        self.conditional.name = self.name + '_conditional'

    def set_tparams(self, excludes=[]):
        print 'Excluding the following parameters from learning: %s' % excludes
        tparams = super(GBN, self).set_tparams()
        tparams.update(**self.posterior.set_tparams())
        tparams.update(**self.conditional.set_tparams())
        tparams.update(**self.prior.set_tparams())
        tparams = OrderedDict((k, v) for k, v in tparams.iteritems()
            if k not in excludes)

        return tparams

    def get_params(self):
        params = self.prior.get_params() + self.conditional.get_params()
        return params

    def get_prior_params(self, *params):
        params = list(params)
        return params[:self.prior.n_params]

    # Latent sampling ---------------------------------------------------------

    def sample_from_prior(self, n_samples=100):
        h, updates = self.prior.sample(n_samples)
        py = self.conditional.feed(h)
        return self.get_center(py), updates

    def visualize_latents(self):
        h0 = self.prior.mu
        py0 = self.conditional.feed(h0)

        sq = T.nlinalg.AllocDiag()(self.prior.log_sigma)
        h = T.exp(sq).astype(floatX) + h0[None, :]
        py = self.conditional.feed(h)

        return py - py0[None, :]

    # Misc --------------------------------------------------------------------

    def get_center(self, p):
        return self.conditional.get_center(p)

    def p_y_given_h(self, h, *params):
        start  = self.prior.n_params
        stop   = start + self.conditional.n_params
        params = params[start:stop]
        return self.conditional.step_feed(h, *params)

    def kl_divergence(self, p, q):
        dim = self.dim_h
        mu_p = _slice(p, 0, dim)
        log_sigma_p = _slice(p, 1, dim)
        mu_q = _slice(q, 0, dim)
        log_sigma_q = _slice(q, 1, dim)

        kl = log_sigma_q - log_sigma_p + 0.5 * (
            (T.exp(2 * log_sigma_p) + (mu_p - mu_q) ** 2) /
            T.exp(2 * log_sigma_q)
            - 1)
        return kl.sum(axis=kl.ndim-1)

    def l2_decay(self, rate):
        rec_l2_cost = self.posterior.get_L2_weight_cost(rate)
        gen_l2_cost = self.conditional.get_L2_weight_cost(rate)

        rval = OrderedDict(
            rec_l2_cost=rec_l2_cost,
            gen_l2_cost=gen_l2_cost,
            cost=rec_l2_cost+gen_l2_cost
        )

        return rval

    def __call__(self, x, y, qk=None, n_posterior_samples=10):
        q0 = self.posterior.feed(x)

        if qk is None:
            qk = q0

        h, updates  = self.posterior.sample(qk, n_samples=n_posterior_samples)
        py          = self.conditional.feed(h)

        log_py_h    = -self.conditional.neg_log_prob(y[None, :, :], py)
        KL_qk_p     = self.prior.kl_divergence(qk)
        KL_qk_q0    = self.kl_divergence(qk, q0)

        log_ph      = -self.prior.neg_log_prob(h)
        log_qh      = -self.posterior.neg_log_prob(h, qk)

        log_p       = (log_sum_exp(log_py_h + log_ph - log_qh, axis=0)
                       - T.log(n_posterior_samples))

        y_energy    = -log_py_h.mean(axis=0)
        p_entropy   = self.prior.entropy()
        q_entropy   = self.posterior.entropy(qk)
        nll         = -log_p

        cost        = (y_energy + KL_qk_p + KL_qk_q0).mean(0)
        lower_bound = -(y_energy + KL_qk_p).mean()

        results = OrderedDict({
            '-log p(x|h)': y_energy.mean(0),
            '-log p(x)': nll.mean(0),
            '-log p(h)': log_ph.mean(),
            '-log q(h)': log_qh.mean(),
            'KL(q_k||p)': KL_qk_p.mean(0),
            'KL(q_k||q_0)': KL_qk_q0.mean(0),
            'H(p)': p_entropy,
            'H(q)': q_entropy.mean(0),
            'lower_bound': lower_bound,
            'cost': cost
        })

        samples = OrderedDict(
            py=py,
            batch_energies=y_energy
        )

        return results, samples


#Deep Gaussian Belief Networks==================================================


class DeepGBN(Layer):
    def __init__(self, dim_in, dim_h, dim_out, n_layers=2,
                 posteriors=None, conditionals=None,
                 z_init=None,
                 name='gbn',
                 **kwargs):

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_h = dim_h

        self.n_layers = n_layers

        self.posteriors = posteriors
        self.conditionals = conditionals

        self.z_init = z_init

        kwargs = init_inference_args(self, **kwargs)
        kwargs = init_weights(self, **kwargs)
        kwargs = init_rngs(self, **kwargs)

        super(DeepGBN, self).__init__(name=name)

    def set_params(self):
        mu = np.zeros((self.dim_h,)).astype(floatX)
        log_sigma = np.zeros((self.dim_h,)).astype(floatX)
        self.params = OrderedDict(mu=mu, log_sigma=log_sigma)

        if self.posteriors is None:
            self.posteriors = [None for _ in xrange(self.n_layers)]
        else:
            assert len(posteriors) == self.n_layers

        if self.conditionals is None:
            self.conditionals = [None for _ in xrange(self.n_layers)]
        else:
            assert len(conditionals) == self.n_layers

        for l in xrange(self.n_layers):
            if l == 0:
                dim_in = self.dim_in
            else:
                dim_in = self.dim_h

            if l == self.n_layers - 1:
                dim_out = self.dim_out
            else:
                dim_out = self.dim_h

            if self.posteriors[l] is None:
                self.posteriors[l] = MLP(
                    dim_in, dim_out, dim_out, 1,
                    rng=self.rng, trng=self.trng,
                    h_act='T.nnet.sigmoid',
                    out_act='lambda x: x')

            if l == 0:
                out_act = 'T.nnet.sigmoid'
            else:
                out_act = 'lambda x: x'

            if self.conditionals[l] is None:
                self.conditionals[l] = MLP(
                    dim_out, dim_out, dim_in, 1,
                    rng=self.rng, trng=self.trng,
                    h_act='T.nnet.sigmoid',
                    out_act=out_act)

            self.posteriors[l].name = self.name + '_posterior%d' % l
            self.conditionals[l].name = self.name + '_conditional%d' % l

    def set_tparams(self, excludes=[]):
        excludes = ['{name}_{key}'.format(name=self.name, key=key)
                    for key in excludes]
        tparams = super(DeepGBN, self).set_tparams()

        for l in xrange(self.n_layers):
            tparams.update(**self.posteriors[l].set_tparams())
            tparams.update(**self.conditionals[l].set_tparams())

        tparams = OrderedDict((k, v) for k, v in tparams.iteritems()
            if k not in excludes)

        return tparams

    def get_params(self):
        params = [self.mu, self.log_sigma]
        for l in xrange(self.n_layers):
            params += self.conditionals[l].get_params()

        return params

    def p_y_given_h(self, h, level, *params):
        start = 2
        for n in xrange(level):
            start += len(self.conditionals[n].get_params())
        end = start + len(self.conditionals[level].get_params())

        params = params[start:end]
        return self.conditionals[level].step_call(h, *params)

    def sample_from_prior(self, n_samples=100):
        p = T.concatenate([self.mu, self.log_sigma])
        h, updates = self.posteriors[-1].sample(p=p, size=(n_samples, self.dim_h))

        for conditional in self.conditionals[::-1]:
            p = conditional(h)
            h = conditional.sample(p)

        return p, updates

    def kl_divergence(self, p, q,
                      entropy_scale=1.0):
        dim = self.dim_h
        mu_p = _slice(p, 0, dim)
        log_sigma_p = _slice(p, 1, dim)
        mu_q = _slice(q, 0, dim)
        log_sigma_q = _slice(q, 1, dim)

        kl = log_sigma_q - log_sigma_p + 0.5 * (
            (T.exp(2 * log_sigma_p) + (mu_p - mu_q) ** 2) /
            T.exp(2 * log_sigma_q)
            - 1)
        return kl.sum(axis=kl.ndim-1)

    def m_step(self, p_hs, y, qs, n_samples=10):
        constants = []

        prior_energy = T.constant(0.).astype(floatX)
        y_energy = T.constant(0.).astype(floatX)
        h_energy = T.constant(0.).astype(floatX)

        mu = self.mu[None, None, :]
        log_sigma = self.log_sigma[None, None, :]
        p_y = concatenate([mu, log_sigma], axis=2)

        for l in xrange(self.n_layers - 1, -1, -1):
            q = qs[l]
            p_h = p_hs[l]
            h_energy += self.kl_divergence(q, p_h).mean()
            prior_energy += self.kl_divergence(q[None, :, :], p_y).mean()

            if n_samples == 0:
                h = mu[None, :, :]
            else:
                h = self.posteriors[l].sample(p=q, size=(n_samples, q.shape[0], q.shape[1] / 2))

            p_y = self.conditional(h)

            if l == 0:
                y_energy += self.conditionals[l].neg_log_prob(y[None, :, :], p_y).mean()
            else:
                y_energy += self.kl_divergence(q[l - 1][None, :, :], p_y).mean()

        return (prior_energy, h_energy, y_energy), constants

    def e_step(self, y, qs, *params):
        prior = concatenate(params[:1], axis=0)
        consider_constant = [y, prior]
        cost = T.constant(0.).astype(floatX)

        for l in xrange(self.n_layers):
            q = qs[l]
            mu_q = _slice(q, 0, self.dim_h)
            log_sigma_q = _slice(q, 1, self.dim_h)

            kl_term = self.kl_divergence(q, prior).mean(axis=0)

            epsilon = self.trng.normal(
                avg=0, std=1.0,
                size=(self.n_inference_samples, mu_q.shape[0], mu_q.shape[1]))

            h = mu_q + epsilon * T.exp(log_sigma_q)
            p = self.p_y_given_h(h, *params)

            if l == 0:
                cond_term = self.conditional.neg_log_prob(y[None, :, :], p).mean(axis=0)
            else:
                cond_term = self.kl_divergence(q[l-1][None, :, :], p)

            cost += (kl_term + cond_term).sum(axis=0)

        grads = theano.grad(cost, wrt=qs, consider_constant=consider_constant)

        return cost, grads

    def step_infer(self, *params):
        raise NotImplementedError()

    def init_infer(self, q):
        raise NotImplementedError()

    def unpack_infer(self, outs):
        raise NotImplementedError()

    def params_infer(self):
        raise NotImplementedError()

    # Momentum
    def _step_momentum(self, y, *params):
        params = list(params)
        qs = params[:self.n_layers]
        l = params[self.n_layers]
        dqs_ = params[1+self.n_layers:1+2*self.n_layers]
        m = params[1+2*self.n_layers]

        params = params[2+2*self.n_layers:]

        cost, grads = self.e_step(y, qs, *params)

        dqs = [(-l * grad + m * dq_).astype(floatX) for dq_, grad in zip(dqs_, grads)]
        qs = [(q + dq).astype(floatX) for q, dq in zip(qs, dqs)]

        l *= self.inference_decay
        return qs + (l,) + dqs + (cost,)

    def _init_momentum(self, qs):
        return [self.inference_rate] + [T.zeros_like(q) for q in qs]

    def _unpack_momentum(self, outs):
        qss = outs[:self.n_layers]
        costs = outs[-1]
        return qss, costs

    def _params_momentum(self):
        return [T.constant(self.momentum).astype('float32')]

    def infer_q(self, x, y, n_inference_steps, q0s=None):
        updates = theano.OrderedUpdates()

        xs, ys = self.init_inputs(x, y, steps=n_inference_steps+1)

        state = xs
        q0s = []
        phs = []

        for l in xrange(self.n_layers):
            ph = self.posteriors[l](state)
            phs.append(ph)
            state = self.posteriors[l].sample(ph)

            if self.z_init == 'recognition_net':
                print 'Starting q0 at recognition net'
                q0 = ph[0]
            else:
                raise NotImplementedError()
                q0 = T.alloc(0., x.shape[0], 2 * self.dim_h).astype(floatX)
            q0s.append(q0)

        seqs = [ys]
        outputs_info = q0s + self.init_infer(q0s) + [None]
        non_seqs = self.params_infer() + self.get_params()

        if n_inference_steps > 0:
            outs, updates_2 = theano.scan(
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

            qss, i_costs = self.unpack_infer(outs)

            for l in xrange(self.n_layers):
                qss[l] = T.concatenate([q0s[l][None, :, :], qss[l]], axis=0)
        else:
            qss = [q0[None, :, :] for q0 in q0s]

        return (phs, xs, ys, qss), updates

    # Inference
    def inference(self, x, y, q0s=None, n_inference_steps=20, n_sampling_steps=None, n_samples=100):
        (phs, xs, ys, qss), updates = self.infer_q(
            x, y, n_inference_steps, q0s=q0s)

        qs = [qs[-1] for qs in qss]
        phs = [ph[0] for ph in phs]

        (prior_energy, h_energy, y_energy), m_constants = self.m_step(
            phs, ys[0], qs, n_samples=n_samples)

        constants = [xs, ys, qs, entropy] + m_constants

        return (xs, ys, qs,
                prior_energy, h_energy, y_energy,
                y_energy_approx, entropy), updates, constants

    def __call__(self, x, y, n_samples=100, n_sampling_steps=None,
                 n_inference_steps=0, end_with_inference=True, sample=False):

        outs = OrderedDict()
        updates = theano.OrderedUpdates()

        q0s = []

        prior = T.concatenate([self.mu[None, :], self.log_sigma[None, :]], axis=1)

        if end_with_inference:
            (_, xs, ys, qss), updates_i = self.infer_q(
                x, y, n_inference_steps,)
            updates.update(updates_i)

            q = [qs[-1] for qs in qss]
        else:
            ys = x.copy()

        if n_samples == 0:
            h = q[None, :, :]
        else:
            h = self.posterior.sample(
                q, size=(n_samples, q.shape[0], q.shape[1] / 2))

        py = self.conditional(h)
        y_energy = self.conditionals[0].neg_log_prob(ys, py).mean(axis=(0, 1))
        kl_term = self.kl_divergence(q, prior).mean(axis=0)

        outs.update(
            py=py,
            lower_bound=(y_energy+kl_term)
        )

        return outs, updates
