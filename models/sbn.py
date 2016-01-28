'''
module of Stochastic Feed Forward Networks
'''

from collections import OrderedDict
import numpy as np
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from layers import Layer
from darn import AutoRegressor, DARN
from distributions import Bernoulli, Gaussian
from mlp import (
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
    _slice
)


def init_momentum_args(model, momentum=0.9, **kwargs):
    model.momentum = momentum
    return kwargs

def init_sgd_args(model, **kwargs):
    return kwargs

def init_inference_args(model,
                        inference_rate=0.1,
                        inference_decay=0.99,
                        entropy_scale=1.0,
                        importance_sampling=False,
                        n_inference_samples=20,
                        inference_scaling=None,
                        inference_method='momentum',
                        alpha=7,
                        center_latent=False,
                        extra_inference_args=dict(),
                        **kwargs):
    model.inference_rate = inference_rate
    model.inference_decay = inference_decay
    model.entropy_scale = entropy_scale
    model.importance_sampling = importance_sampling
    model.inference_scaling = inference_scaling
    model.n_inference_samples = n_inference_samples
    model.alpha = alpha
    model.center_latent = center_latent

    if inference_method == 'sgd':
        model.step_infer = model._step_sgd
        model.init_infer = model._init_sgd
        model.unpack_infer = model._unpack_sgd
        model.params_infer = model._params_sgd
        kwargs = init_sgd_args(model, **extra_inference_args)
    elif inference_method == 'momentum':
        model.step_infer = model._step_momentum
        model.init_infer = model._init_momentum
        model.unpack_infer = model._unpack_momentum
        model.params_infer = model._params_momentum
        kwargs = init_momentum_args(model, **extra_inference_args)
    elif inference_method == 'momentum_straight_through':
        model.step_infer = model._step_momentum_st
        model.init_infer = model._init_momentum
        model.unpack_infer = model._unpack_momentum
        model.params_infer = model._params_momentum
        kwargs = init_momentum_args(model, **extra_inference_args)
    elif inference_method == 'adaptive':
        model.step_infer = model._step_adapt
        model.init_infer = model._init_adapt
        model.unpack_infer = model._unpack_adapt
        model.params_infer = model._params_adapt
        model.strict = False
    else:
        raise ValueError('Inference method <%s> not supported' % inference_method)

    return kwargs

def _sample(p, size=None, trng=None):
    if size is None:
        size = p.shape
    return trng.binomial(p=p, size=size, n=1, dtype=p.dtype)

def _noise(x, amount=0.1, size=None, trng=None):
    if size is None:
        size = x.shape
    return x * (1 - trng.binomial(p=amount, size=size, n=1, dtype=x.dtype))

def set_input(x, mode, trng=None):
    if mode == 'sample':
        x = _sample(x, trng=trng)
    elif mode == 'noise':
        x = _noise(x, trng=trng)
    elif mode is None:
        pass
    else:
        raise ValueError('% not supported' % mode)
    return x

def unpack(dim_in=None,
           dim_h=None,
           z_init=None,
           recognition_net=None,
           generation_net=None,
           prior=None,
           dataset=None,
           dataset_args=None,
           n_inference_steps=None,
           n_inference_samples=None,
           inference_method=None,
           inference_rate=None,
           input_mode=None,
           extra_inference_args=dict(),
           **model_args):
    '''
    Function to unpack pretrained model into fresh SFFN class.
    '''
    from gbn import GaussianBeliefNet as GBN

    kwargs = dict(
        inference_method=inference_method,
        inference_rate=inference_rate,
        n_inference_steps=n_inference_steps,
        z_init=z_init,
        n_inference_samples=n_inference_samples,
        extra_inference_args=extra_inference_args,
    )

    dim_h = int(dim_h)
    dataset_args = dataset_args[()]
    recognition_net = recognition_net[()]
    generation_net = generation_net[()]
    if prior == 'logistic':
        out_act = 'T.nnet.sigmoid'
        prior_model = Bernoulli(dim_h)
        C = SigmoidBeliefNetwork
    elif prior == 'darn':
        out_act = 'T.nnet.sigmoid'
        prior_model = AutoRegressor(dim_h)
        C = SigmoidBeliefNetwork
    elif prior == 'gaussian':
        out_act = 'lambda x: x'
        prior_model = Gaussian(dim_h)
        C = GBN 
    else:
        raise ValueError('%s prior not known' % prior)

    models = []

    mlps = SigmoidBeliefNetwork.mlp_factory(recognition_net=recognition_net,
                                            generation_net=generation_net)

    models += mlps.values()
    models.append(prior_model)

    kwargs.update(**mlps)
    kwargs['prior'] = prior_model
    model = C(dim_in, dim_h, **kwargs)
    models.append(model)
    return models, model_args, dict(
        dataset=dataset,
        dataset_args=dataset_args
    )


class SigmoidBeliefNetwork(Layer):
    def __init__(self, dim_in, dim_h,
                 posterior=None, conditional=None,
                 prior=None,
                 z_init=None,
                 name='sbn',
                 **kwargs):

        self.dim_in = dim_in
        self.dim_h = dim_h

        self.posterior = posterior
        self.conditional = conditional
        self.prior = prior

        self.z_init = z_init

        kwargs = init_inference_args(self, **kwargs)
        kwargs = init_weights(self, **kwargs)
        kwargs = init_rngs(self, **kwargs)

        super(SigmoidBeliefNetwork, self).__init__(name=name)

    @staticmethod
    def mlp_factory(recognition_net=None, generation_net=None):
        mlps = {}

        if recognition_net is not None:
            t = recognition_net.get('type', None)
            print recognition_net
            if t is None:
                posterior = MLP.factory(**recognition_net)
            elif t == 'darn':
                posterior = DARN.factory(**recognition_net)
            else:
                raise ValueError(t)
            mlps['posterior'] = posterior

        if generation_net is not None:
            t = generation_net.get('type', None)
            if t is None:
                conditional = MLP.factory(**generation_net)
            elif t == 'darn':
                conditional = DARN.factory(**generation_net)
            elif t == 'MMMLP':
                conditional = MultiModalMLP.factory(**generation_net)
            else:
                raise ValueError(t)
            mlps['conditional'] = conditional

        return mlps

    def set_params(self):
        self.params = OrderedDict()

        if self.prior is None:
            self.prior = Bernoulli(self.dim_h)
        elif isinstance(self.prior, Gaussian):
            raise NotImplementedError('Gaussian prior not supported here ATM. '
                                      'Try gbn.py')

        if self.posterior is None:
            self.posterior = MLP(self.dim_in, self.dim_h, self.dim_h, 1,
                                 rng=self.rng, trng=self.trng,
                                 h_act='T.nnet.sigmoid',
                                 out_act='T.nnet.sigmoid')
        elif isinstance(self.posterior, DARN):
            raise ValueError('DARN posterior not supported ATM')

        if self.conditional is None:
            self.conditional = MLP(self.dim_h, self.dim_h, self.dim_in, 1,
                                   rng=self.rng, trng=self.trng,
                                   h_act='T.nnet.sigmoid',
                                   out_act='T.nnet.sigmoid')

        self.posterior.name = self.name + '_posterior'
        self.conditional.name = self.name + '_conditional'

    def set_tparams(self, excludes=[]):
        excludes = ['{name}_{key}'.format(name=self.name, key=key)
                    for key in excludes]
        tparams = super(SigmoidBeliefNetwork, self).set_tparams()
        tparams.update(**self.posterior.set_tparams())
        tparams.update(**self.conditional.set_tparams())
        tparams.update(**self.prior.set_tparams())
        tparams = OrderedDict((k, v) for k, v in tparams.iteritems()
            if k not in excludes)

        return tparams

    # Fetch params -------------------------------------------------------------

    def get_params(self):
        params = self.prior.get_params() + self.conditional.get_params() + self.posterior.get_params()
        return params

    def get_prior_params(self, *params):
        params = list(params)
        return params[:self.prior.n_params]

    def get_posterior_params(self, *params):
        params = list(params)
        start = self.prior.n_params + self.conditional.n_params
        stop = start + self.posterior.n_params
        return params[start:stop]

    # Latent sampling ---------------------------------------------------------

    def sample_from_prior(self, n_samples=99):
        h, updates = self.prior.sample(n_samples)
        #p = T.nnet.sigmoid(self.prior.z)
        #r = self.trng.uniform((n_samples, self.prior.dim), dtype=floatX)
        #h = (r <= p[None, :]).astype(floatX)
        #updates = theano.OrderedUpdates()
        return self.conditional(h), updates

    def generate_from_latent(self, h):
        py = self.conditional(h)
        prob = self.conditional.prob(py)
        return prob

    # Misc --------------------------------------------------------------------

    def get_center(self, p):
        return self.conditional.prob(p)

    def log_marginal(self, y, h, py, q):
        log_py_h = -self.conditional.neg_log_prob(y, py)
        log_ph   = -self.prior.neg_log_prob(h)
        log_qh   = -self.posterior.neg_log_prob(h, q)
        assert log_py_h.ndim == log_ph.ndim == log_qh.ndim

        log_p     = log_py_h + log_ph - log_qh
        log_p_max = T.max(log_p, axis=0, keepdims=True)
        w         = T.exp(log_p - log_p_max)

        return (T.log(w.mean(axis=0, keepdims=True)) + log_p_max).mean()

    def p_y_given_h(self, h, *params):
        start  = self.prior.n_params
        stop   = start + self.conditional.n_params
        params = params[start:stop]
        return self.conditional.step_call(h, *params)

    # Inference ---------------------------------------------------------------
    # Note: only AIR here

    def step_infer(self, *params):
        raise NotImplementedError()
    def init_infer(self, z):
        raise NotImplementedError()
    def unpack_infer(self, outs):
        raise NotImplementedError()
    def params_infer(self):
        raise NotImplementedError()

    # Importance Sampling
    def _step_adapt(self, r, q, y, q0, *params):
        prior_params = self.get_prior_params(*params)

        h        = (r <= q[None, :, :]).astype(floatX)
        py       = self.p_y_given_h(h, *params)
        log_py_h = -self.conditional.neg_log_prob(y[None, :, :], py)
        log_ph   = -self.prior.step_neg_log_prob(h, *prior_params)
        log_qh   = -self.posterior.neg_log_prob(h, q[None, :, :])

        assert log_py_h.ndim == log_ph.ndim == log_qh.ndim

        log_p     = log_py_h + log_ph - log_qh
        log_p_max = T.max(log_p, axis=0, keepdims=True)

        w       = T.exp(log_p - log_p_max)
        w_tilde = w / w.sum(axis=0, keepdims=True)
        cost    = -log_p.mean() #-T.log(w).mean()

        q_ = (w_tilde[:, :, None] * h).sum(axis=0)
        q  = self.inference_rate * q_ + (1 - self.inference_rate) * q

        return q, cost

    def _init_adapt(self, q):
        return []

    def _unpack_adapt(self, q0, outs):
        if outs is not None:
            qs, costs = outs
            if qs.ndim == 2:
                qs = concatenate([q0[None, :, :], qs[None, :, :]], axis=0)
                costs = [costs]
            else:
                qs = concatenate([q0[None, :, :], qs])

        else:
            qs = q0[None, :, :]
            costs = [T.constant(0.).astype(floatX)]
        return qs, costs

    def _params_adapt(self):
        return []

    # Learning -----------------------------------------------------------------

    def m_step(self, x, y, qk, n_samples=10):
        constants = []
        updates = theano.OrderedUpdates()

        q0  = self.posterior(x)
        q0_c = T.zeros_like(q0) + q0
        constants.append(q0_c)
        r   = self.trng.uniform((n_samples, y.shape[0], self.dim_h), dtype=floatX)
        h   = (r <= q0[None, :, :]).astype(floatX)
        py  = self.conditional(h)

        if self.prior.must_sample:
            prior_energy = self.prior.neg_log_prob(h).mean(axis=0)
        else:
            prior_energy = self.prior.neg_log_prob(q0_c)

        entropy  = self.posterior.entropy(qk)
        h_energy = self.posterior.neg_log_prob(qk, q0)
        y_energy = self.conditional.neg_log_prob(y[None, :, :], py).mean(axis=0)

        assert prior_energy.ndim == h_energy.ndim == entropy.ndim == y_energy.ndim

        return (prior_energy, h_energy,
                y_energy, entropy), constants, updates

    def infer_q(self, x, y, n_inference_steps):
        updates = theano.OrderedUpdates()

        q0 = self.posterior(x)
        rs = self.trng.uniform(
            (n_inference_steps,
             self.n_inference_samples,
             y.shape[0],
             self.dim_h),
            dtype=floatX)

        seqs = [rs]
        outputs_info = [q0, None]
        non_seqs = [y, q0] + self.params_infer() + self.get_params()

        print ('Doing %d inference steps and a rate of %.2f with %d '
               'inference samples'
               % (n_inference_steps, self.inference_rate, self.n_inference_samples))

        if isinstance(n_inference_steps, T.TensorVariable) or n_inference_steps > 1:
            outs, updates_2 = theano.scan(
                self.step_infer,
                sequences=seqs,
                outputs_info=outputs_info,
                non_sequences=non_seqs,
                name=self.name + '_infer',
                n_steps=n_inference_steps,
                profile=tools.profile,
                strict=False
            )
            updates.update(updates_2)

            qs, i_costs = self.unpack_infer(q0, outs)

        elif n_inference_steps == 1:
            inps = [rs[0]] + outputs_info[:-1] + non_seqs
            outs = self.step_infer(*inps)
            qs, i_costs = self.unpack_infer(q0, outs)

        elif n_inference_steps == 0:
            qs, i_costs = self.unpack_infer(q0, None)

        return (qs, rs), updates

    def inference(self, x, y, n_inference_steps=20, n_samples=100, pass_gradients=False):
        (qs, i_costs), updates = self.infer_q(x, y, n_inference_steps)
        qk = qs[-1]

        (prior_energy, h_energy, y_energy, entropy), m_constants, updates_s = self.m_step(
            x, y, qk, n_samples=n_samples)
        updates.update(updates_s)

        constants = [qk, entropy] + m_constants

        return (qk, prior_energy, h_energy, y_energy, entropy), updates, constants

    # Sampling and test --------------------------------------------------------

    def __call__(self, x, y, n_samples=100, n_inference_steps=0,
                 calculate_log_marginal=False, stride=10):
        outs = OrderedDict()

        (qs, i_costs), updates = self.infer_q(x, y, n_inference_steps)

        if n_inference_steps > stride and stride != 0:
            steps = [0, 1] + range(stride, n_inference_steps, stride)
            steps = steps[:-1] + [n_inference_steps - 1]
        elif n_inference_steps > 0 and n_inference_steps != 1:
            steps = [0, 1, n_inference_steps - 1]
        elif n_inference_steps == 1:
            steps = [0, 1]
        else:
            steps = [0]

        lower_bounds = []
        nlls = []
        pys = []
        energies = []
        prior_terms = []
        entropy_terms = []

        for i in steps:
            q  = qs[i]
            r  = self.trng.uniform((n_samples, y.shape[0], self.dim_h), dtype=floatX)
            h  = (r <= q[None, :, :]).astype(floatX)
            py = self.conditional(h)
            pys.append(py)
            y_energy_b = self.conditional.neg_log_prob(y[None, :, :], py).mean(axis=0)
            energies.append(y_energy_b)

            if self.prior.must_sample:
                prior_term = self.prior.neg_log_prob(h).mean(axis=0)
            else:
                prior_term = self.prior.neg_log_prob(q)

            cond_term    = self.conditional.neg_log_prob(y[None, :, :], py).mean(axis=0)
            entropy_term = self.posterior.entropy(q)

            prior_terms.append(prior_term.mean())
            entropy_terms.append(entropy_term.mean())

            assert prior_term.ndim == entropy_term.ndim

            kl_term = (prior_term - entropy_term)
            lower_bounds.append((cond_term + kl_term).mean())

            if calculate_log_marginal:
                nll = -self.log_marginal(y[None, :, :], h, py, q[None, :, :])
                nlls.append(nll)

        outs.update(
            i_costs=i_costs[steps],
            prior_terms=prior_terms,
            entropy_terms=entropy_terms,
            energies=energies,
            py0=pys[0],
            py=pys[-1],
            pys=pys,
            lower_bound=lower_bounds[-1],
            lower_bounds=lower_bounds,
            inference_cost=(lower_bounds[0] - lower_bounds[-1])
        )

        if calculate_log_marginal:
            outs.update(nll=nlls[-1], nlls=nlls)

        return outs, updates
