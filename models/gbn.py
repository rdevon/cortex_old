'''
Module of Gaussian Belief Networks
'''

from collections import OrderedDict
import numpy as np
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from layers import (
    Layer,
    MLP
)
from sbn import init_inference_args
from utils import tools
from utils.tools import (
    concatenate,
    init_rngs,
    init_weights,
    log_mean_exp,
    log_sum_exp,
    logit,
    _slice
)

norm_weight = tools.norm_weight
ortho_weight = tools.ortho_weight
floatX = 'float32' # theano.config.floatX
pi = theano.shared(np.pi).astype('float32')


class GaussianBeliefNet(Layer):
    def __init__(self, dim_in, dim_h,
                 posterior=None, conditional=None,
                 z_init=None,
                 name='gbn',
                 **kwargs):

        self.strict = True
        self.dim_in = dim_in
        self.dim_h = dim_h

        self.posterior = posterior
        self.conditional = conditional

        self.z_init = z_init

        kwargs = init_inference_args(self, **kwargs)
        kwargs = init_weights(self, **kwargs)
        kwargs = init_rngs(self, **kwargs)

        super(GaussianBeliefNet, self).__init__(name=name)

    def set_params(self):
        mu = np.zeros((self.dim_h,)).astype(floatX)
        log_sigma = np.zeros((self.dim_h,)).astype(floatX)

        self.params = OrderedDict(
            mu=mu, log_sigma=log_sigma)

        if self.posterior is None:
            self.posterior = MLP(self.dim_in, self.dim_h, self.dim_h, 1,
                                 rng=self.rng, trng=self.trng,
                                 h_act='T.nnet.sigmoid',
                                 out_act='lambda x: x')
        if self.conditional is None:
            self.conditional = MLP(self.dim_h, self.dim_h, self.dim_in, 1,
                                   rng=self.rng, trng=self.trng,
                                   h_act='T.nnet.sigmoid',
                                   out_act='T.nnet.sigmoid')

        self.posterior.name = self.name + '_posterior'
        self.conditional.name = self.name + '_conditional'

    def set_tparams(self, excludes=[]):
        excludes = [ex for ex in excludes if ex in self.params.keys()]
        print 'Excluding the following parameters from learning: %s' % excludes
        excludes = ['{name}_{key}'.format(name=self.name, key=key)
                    for key in excludes]
        tparams = super(GaussianBeliefNet, self).set_tparams()
        tparams.update(**self.posterior.set_tparams())
        tparams.update(**self.conditional.set_tparams())
        tparams = OrderedDict((k, v) for k, v in tparams.iteritems()
            if k not in excludes)

        return tparams

    def get_params(self):
        params = [self.mu, self.log_sigma] + self.conditional.get_params()
        return params

    def p_y_given_h(self, h, *params):
        params = params[2:]
        return self.conditional.step_call(h, *params)

    def sample_from_prior(self, n_samples=100):
        p = T.concatenate([self.mu, self.log_sigma])
        h = self.posterior.sample(p=p, size=(n_samples, self.dim_h))
        py = self.conditional(h)
        return py

    def importance_weights(self, y, h, py, q, prior, normalize=True):
        y_energy = self.conditional.neg_log_prob(y, py)
        prior_energy = self.posterior.neg_log_prob(h, prior)
        entropy_term = self.posterior.neg_log_prob(h, q)

        log_p = -y_energy + entropy_term - prior_energy
        log_p_max = T.max(log_p, axis=0, keepdims=True)
        w = T.exp(log_p - log_p_max)

        if normalize:
            w /= w.sum(axis=0, keepdims=True)

        return w

    def log_marginal(self, y, h, py, q, prior):
        '''
        Computes the negative log marginal.
        '''

        y_energy = self.conditional.neg_log_prob(y, py)
        prior_energy = self.posterior.neg_log_prob(h, prior)
        entropy_term = self.posterior.neg_log_prob(h, q)

        log_p = -y_energy - prior_energy + entropy_term
        log_p_max = T.max(log_p, axis=0, keepdims=True)
        w = T.exp(log_p - log_p_max)

        return (T.log(T.mean(w, axis=0, keepdims=True)) + log_p_max).mean()

    def kl_divergence(self, p, q):
        '''
        Computes the negative KL divergence.
        '''

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

    def e_step(self, y, epsilon, q, *params):
        '''
        E step for IRVI.
        '''
        prior = concatenate([params[0][None, :], params[1][None, :]], axis=1)

        mu_q = _slice(q, 0, self.dim_h)
        log_sigma_q = _slice(q, 1, self.dim_h)
        h = mu_q[None, :, :] + epsilon * T.exp(log_sigma_q[None, :, :])
        py = self.p_y_given_h(h, *params)

        consider_constant = [y] + list(params)
        cond_term = self.conditional.neg_log_prob(y[None, :, :], py).mean(axis=0)

        kl_term = self.kl_divergence(q, prior)

        cost = (cond_term + kl_term).sum(axis=0)
        grad = theano.grad(cost, wrt=q, consider_constant=consider_constant)

        return cost, grad

    def m_step(self, x, y, q, n_samples=10):
        '''
        M step for IRVI in GBNs.
        '''

        constants = []
        prior = T.concatenate([self.mu[None, :], self.log_sigma[None, :]], axis=1)
        p_h = self.posterior(x)
        epsilon = self.trng.normal(avg=0, std=1.0,
                                   size=(n_samples, q.shape[0], q.shape[1] // 2),
                                   dtype=x.dtype)
        #mu_q = _slice(q, 0, self.dim_h)
        #log_sigma_q = _slice(q, 1, self.dim_h)
        #h = mu_q[None, :, :] + epsilon * T.exp(log_sigma_q[None, :, :])
        h = self.posterior.sample(p=q, size=(n_samples, q.shape[0], q.shape[1] // 2))
        p_y = self.conditional(h)

        y_energy = self.conditional.neg_log_prob(y[None, :, :], p_y).mean()
        prior_energy = self.kl_divergence(q, prior).mean()
        h_energy = self.kl_divergence(q, p_h).mean()
        entropy = self.posterior.entropy(q).mean()

        return (prior_energy, h_energy, y_energy, entropy), constants

    def step_infer(self, *params):
        raise NotImplementedError()

    def init_infer(self, q):
        raise NotImplementedError()

    def unpack_infer(self, outs):
        raise NotImplementedError()

    def params_infer(self):
        raise NotImplementedError()

    # Momentum
    def _step_momentum(self, y, epsilon, q, dq_, m, *params):
        l = self.inference_rate
        cost, grad = self.e_step(y, epsilon, q, *params)
        dq = (-l * grad + m * dq_).astype(floatX)
        q = (q + dq).astype(floatX)
        return q, dq, cost

    def _init_momentum(self, q):
        return [T.zeros_like(q)]

    def _unpack_momentum(self, outs):
        qs, dqs, costs = outs
        return qs, costs

    def _params_momentum(self):
        return [T.constant(self.momentum).astype('float32')]

    def init_variational_inference(self, ph):
        if self.z_init == 'recognition_net':
            print 'Starting q0 at recognition net'
            q0 = ph
        else:
            q0 = T.alloc(0., x.shape[0], 2 * self.dim_h).astype(floatX)

        return q0

    def infer_q(self, x, y, n_inference_steps):
        updates = theano.OrderedUpdates()

        ys = T.alloc(0., n_inference_steps + 1, y.shape[0], y.shape[1]) + y[None, :, :]
        ph = self.posterior(x)
        q0 = self.init_variational_inference(ph)

        epsilons = self.trng.normal(
            avg=0, std=1.0,
            size=(n_inference_steps, self.n_inference_samples,
                  x.shape[0], self.dim_h),
            dtype=x.dtype)

        seqs = [ys, epsilons]
        outputs_info = [q0] + self.init_infer(q0) + [None]
        non_seqs = self.params_infer() + self.get_params()

        print ('Doing %d inference steps and a rate of %.5f with %d '
               'inference samples'
               % (n_inference_steps, self.inference_rate, self.n_inference_samples))

        if isinstance(n_inference_steps, T.TensorVariable) or n_inference_steps > 1:
            outs, updates_2 = theano.scan(
                self.step_infer,
                sequences=seqs,
                outputs_info=outputs_info,
                non_sequences=non_seqs,
                name=tools._p(self.name, 'infer'),
                n_steps=n_inference_steps,
                profile=tools.profile,
                strict=False
            )
            updates.update(updates_2)

            qs, i_costs = self.unpack_infer(outs)

        elif n_inference_steps == 1:
            inps = [ys[0], epsilons[0]] + outputs_info[:-1] + non_seqs
            outs = self.step_infer(*inps)
            q, i_cost = self.unpack_infer(outs)
            qs = q[None, :, :]
            i_costs = [i_cost]

        elif n_inference_steps == 0:
            print 'No inference steps. VAE'
            q = q0
            qs = q[None, :, :]
            i_costs = [T.constant(0.).astype(floatX)]

        return (qs, i_costs), updates

    # Inference
    def inference(self, x, y, n_inference_steps=20, n_sampling_steps=None, n_samples=100):
        (qs, _), updates = self.infer_q(x, y, n_inference_steps)
        q = qs[-1]

        (prior_energy, h_energy, y_energy, entropy), m_constants = self.m_step(
            x, y, q, n_samples=n_samples)

        constants = [entropy] + m_constants
        if n_inference_steps > 0:
            constants.append(qs)

        return (q, prior_energy, h_energy, y_energy, entropy), updates, constants

    def __call__(self, x, y, n_samples=100, n_inference_steps=0,
                 calculate_log_marginal=False):

        outs = OrderedDict()
        prior = T.concatenate([self.mu[None, :], self.log_sigma[None, :]], axis=1)

        (qs, i_costs), updates = self.infer_q(x, y, n_inference_steps)

        if n_inference_steps > 10:
            steps = range(n_inference_steps // 10, n_inference_steps + 1, n_inference_steps // 10)
            steps = steps[:-1] + [n_inference_steps]
        elif n_inference_steps > 0:
            steps = [0, n_inference_steps]
        else:
            steps = [0]

        lower_bounds = []
        nlls = []
        for i in steps:
            q = qs[i-1]

            h = self.posterior.sample(
                q, size=(n_samples, q.shape[0], q.shape[1] / 2))

            py = self.conditional(h)
            y_energy = self.conditional.neg_log_prob(y[None, :, :], py).mean(axis=(0, 1))
            kl_term = self.kl_divergence(q, prior).mean(axis=0)
            lower_bound = y_energy + kl_term
            lower_bounds.append(lower_bound)

            if calculate_log_marginal:
                nll = -self.log_marginal(y[None, :, :], h, py, q[None, :, :], prior[None, :, :])
                nlls.append(nll)

        outs.update(
            py=py,
            lower_bound=lower_bounds[-1],
            lower_bounds=lower_bounds,
            inference_cost=(lower_bounds[0] - lower_bounds[-1])
        )

        if calculate_log_marginal:
            outs.update(nll=nlls[-1], nlls=nlls)

        return outs, updates


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
        h = self.posteriors[-1].sample(p=p, size=(n_samples, self.dim_h))

        for conditional in self.conditionals[::-1]:
            p = conditional(h)
            h = conditional.sample(p)

        return p

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
                 n_inference_steps=0, end_with_inference=True):

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
