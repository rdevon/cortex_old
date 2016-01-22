'''
module of Stochastic Feed Forward Networks
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
        model.init_variational_params = model._init_variational_params_adapt
    elif inference_method == 'adaptive_assign':
        model.step_infer = model._step_adapt_assign
        model.init_infer = model._init_adapt
        model.unpack_infer = model._unpack_adapt
        model.params_infer = model._params_adapt
        model.strict = False
        model.init_variational_params = model._init_variational_params_adapt
    elif inference_method == 'momentum_then_adapt':
        model.step_infer = model._step_momentum_then_adapt
        model.init_infer = model._init_momentum
        model.unpack_infer = model._unpack_momentum_then_adapt
        model.params_infer = model._params_momentum
        model.init_variational_params = model._init_variational_params_adapt
        model.strict = False
        kwargs = init_momentum_args(model, **extra_inference_args)
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
        prior=prior,
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
    elif prior == 'gaussian':
        out_act = 'lambda x: x'
    else:
        raise ValueError('%s prior not known' % prior)

    models = []

    mlps = SigmoidBeliefNetwork.mlp_factory(recognition_net=recognition_net,
                           generation_net=generation_net)

    models += mlps.values()

    if prior == 'logistic':
        C = SigmoidBeliefNetwork
    elif prior == 'gaussian':
        C = GBN
    else:
        raise ValueError('%s prior not known' % prior)

    kwargs.update(**mlps)
    model = C(dim_in, dim_h, **kwargs)
    models.append(model)
    return models, model_args, dict(
        dataset=dataset,
        dataset_args=dataset_args
    )


class SigmoidBeliefNetwork(Layer):
    def __init__(self, dim_in, dim_h,
                 posterior=None, conditional=None,
                 z_init=None,
                 name='sbn',
                 **kwargs):

        self.dim_in = dim_in
        self.dim_h = dim_h

        self.posterior = posterior
        self.conditional = conditional

        self.z_init = z_init

        kwargs = init_inference_args(self, **kwargs)
        kwargs = init_weights(self, **kwargs)
        kwargs = init_rngs(self, **kwargs)

        super(SigmoidBeliefNetwork, self).__init__(name=name)

    @staticmethod
    def mlp_factory(recognition_net=None, generation_net=None):
        mlps = {}

        if recognition_net is not None:
            assert not 'type' in recognition_net.keys()
            posterior = MLP.factory(**recognition_net)
            mlps['posterior'] = posterior

        if generation_net is not None:
            t = generation_net.get('type', None)
            if t is None:
                conditional = MLP.factory(**generation_net)
            elif t == 'MMMLP':
                conditional = MultiModalMLP.factory(**generation_net)
            else:
                raise ValueError()
            mlps['conditional'] = conditional

        return mlps

    def set_params(self):
        z = np.zeros((self.dim_h,)).astype(floatX)
        inference_scale_factor = np.float32(1.0)

        self.params = OrderedDict(z=z, inference_scale_factor=inference_scale_factor)

        if self.posterior is None:
            self.posterior = MLP(self.dim_in, self.dim_h, self.dim_h, 1,
                                 rng=self.rng, trng=self.trng,
                                 h_act='T.nnet.sigmoid',
                                 out_act='T.nnet.sigmoid')
        if self.conditional is None:
            self.conditional = MLP(self.dim_h, self.dim_h, self.dim_in, 1,
                                   rng=self.rng, trng=self.trng,
                                   h_act='T.nnet.sigmoid',
                                   out_act='T.nnet.sigmoid')

        self.posterior.name = self.name + '_posterior'
        self.conditional.name = self.name + '_conditional'

    def set_tparams(self, excludes=[]):
        excludes.append('inference_scale_factor')
        excludes = ['{name}_{key}'.format(name=self.name, key=key)
                    for key in excludes]
        tparams = super(SigmoidBeliefNetwork, self).set_tparams()
        tparams.update(**self.posterior.set_tparams())
        tparams.update(**self.conditional.set_tparams())
        tparams = OrderedDict((k, v) for k, v in tparams.iteritems()
            if k not in excludes)

        return tparams

    def get_params(self):
        params = [self.z] + self.conditional.get_params() + self.posterior.get_params() + [self.inference_scale_factor]
        return params

    def p_y_given_h(self, h, *params):
        params = params[1:1+len(self.conditional.get_params())]
        return self.conditional.step_call(h, *params)

    def sample_from_prior(self, n_samples=100):
        p = T.nnet.sigmoid(self.z)
        h = self.posterior.sample(p=p, size=(n_samples, self.dim_h))
        py = self.conditional(h)
        return self.get_center(py)

    def generate_from_latent(self, h):
        py = self.conditional(h)
        prob = self.conditional.prob(py)
        return prob

    def get_center(self, p):
        return self.conditional.prob(p)

    def importance_weights(self, y, h, py, q, prior, normalize=True):
        y_energy = self.conditional.neg_log_prob(y, py)
        prior_energy = self.posterior.neg_log_prob(h, prior)
        entropy_term = self.posterior.neg_log_prob(h, q)

        log_p = -y_energy - prior_energy + entropy_term
        log_p_max = T.max(log_p, axis=0, keepdims=True)
        w = T.exp(log_p - log_p_max)

        if normalize:
            w = w / w.sum(axis=0, keepdims=True)

        return w

    def log_marginal(self, y, h, py, q, prior):
        y_energy = self.conditional.neg_log_prob(y, py)
        prior_energy = self.posterior.neg_log_prob(h, prior)
        entropy_term = self.posterior.neg_log_prob(h, q)

        log_p = -y_energy - prior_energy + entropy_term
        log_p_max = T.max(log_p, axis=0, keepdims=True)
        w = T.exp(log_p - log_p_max)

        return (T.log(w.mean(axis=0, keepdims=True)) + log_p_max).mean()

    def kl_divergence(self, p, q):
        entropy_term = self.posterior.entropy(p)
        prior_term = self.posterior.neg_log_prob(p, q)
        return prior_term - entropy_term

    def e_step(self, y, z, *params):
        prior = T.nnet.sigmoid(params[0])
        q = T.nnet.sigmoid(z)
        py = self.p_y_given_h(q, *params)

        consider_constant = [y, prior]
        cond_term = self.conditional.neg_log_prob(y, py)

        kl_term = self.kl_divergence(q, prior[None, :])
        cost = (cond_term + kl_term).sum(axis=0)

        grad = theano.grad(cost, wrt=z, consider_constant=consider_constant)

        return cost, grad

    def m_step(self, x, y, z, n_samples=10):
        constants = []
        q = T.nnet.sigmoid(z)
        prior = T.nnet.sigmoid(self.z)
        p_h = self.posterior(x)

        h = self.posterior.sample(
            q, size=(n_samples, q.shape[0], q.shape[1]))
        py = self.conditional(h)

        entropy = self.posterior.entropy(q).mean()

        prior_energy = self.posterior.neg_log_prob(q, prior[None, :]).mean()
        y_energy = self.conditional.neg_log_prob(y[None, :, :], py).mean()
        h_energy = self.posterior.neg_log_prob(q, p_h).mean()

        return (prior_energy, h_energy, y_energy, entropy), constants

    def step_infer(self, *params):
        raise NotImplementedError()
    def init_infer(self, z):
        raise NotImplementedError()
    def unpack_infer(self, outs):
        raise NotImplementedError()
    def params_infer(self):
        raise NotImplementedError()

    # Importance Sampling
    def _step_adapt(self, y, q, *params):
        prior = T.nnet.sigmoid(params[0])
        h = self.posterior.sample(
            q, size=(self.n_inference_samples, q.shape[0], q.shape[1]))

        py = self.p_y_given_h(h, *params)

        y_energy = self.conditional.neg_log_prob(y[None, :, :], py)
        prior_energy = self.posterior.neg_log_prob(h, prior[None, None, :])
        entropy_term = self.posterior.neg_log_prob(h, q[None, :, :])

        log_p = -y_energy - prior_energy + entropy_term
        log_p_max = T.max(log_p, axis=0, keepdims=True)
        w = T.exp(log_p - log_p_max)

        w_tilde = w / w.sum(axis=0, keepdims=True)

        cost = (log_p - log_p_max).mean()
        q_ = (w_tilde[:, :, None] * h).sum(axis=0)
        q = self.inference_rate * q_ + (1 - self.inference_rate) * q

        return q, cost

    def _init_adapt(self, q):
        return []

    def _init_variational_params_adapt(self, p_h_logit):
        if self.z_init == 'recognition_net':
            print 'Starting z0 at recognition net'
            q0 = T.nnet.sigmoid(p_h_logit)
        else:
            q0 = T.alloc(0.5, p_h_logit.shape[0], self.dim_h).astype(floatX)

        return q0

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
        return logit(qs), costs

    def _params_adapt(self):
        return []

    # SGD
    def _step_sgd(self, y, z, l, *params):
        cost, grad = self.e_step(y, z, *params)
        z = (z - l * grad).astype(floatX)
        l *= self.inference_decay
        return z, l, cost

    def _init_sgd(self, ph, y, z):
        return [self.inference_rate]

    def _unpack_sgd(self, outs):
        zs, ls, costs = outs
        return zs, costs

    def _params_sgd(self):
        return []

    # Momentum
    def _step_momentum(self, y, z, dz_, m, *params):
        l = self.inference_rate
        cost, grad = self.e_step(y, z, *params)
        dz = (-l * grad + m * dz_).astype(floatX)
        z = (z + dz).astype(floatX)
        return z, dz, cost

    def _init_momentum(self, z):
        return [T.zeros_like(z)]

    def _unpack_momentum(self, z0, outs):
        if outs is not None:
            zs, dzs, costs = outs
            if zs.ndim == 2:
                zs = zs[None, :, :]
                costs = [costs]
            zs = concatenate([z0[None, :, :], zs])
        else:
            zs = z0[None, :, :]
            costs = [T.constant(0.).astype(floatX)]
        return zs, costs

    def _unpack_momentum_then_adapt(self, outs):
        qs, ls, dqs, costs = outs
        return logit(qs), costs

    def _params_momentum(self):
        return [T.constant(self.momentum).astype('float32')]

    def init_variational_params(self, p_h_logit):
        if self.z_init == 'recognition_net':
            print 'Starting z0 at recognition net'
            z0 = p_h_logit
        else:
            z0 = T.alloc(0., p_h_logit.shape[0], self.dim_h).astype(floatX)

        return z0

    def infer_q(self, x, y, n_inference_steps):
        updates = theano.OrderedUpdates()

        ys = T.alloc(0., n_inference_steps + 1, y.shape[0], y.shape[1]) + y[None, :, :]
        p_h_logit = self.posterior(x, return_preact=True)
        z0 = self.init_variational_params(p_h_logit)

        seqs = [ys]
        outputs_info = [z0] + self.init_infer(z0) + [None]
        non_seqs = self.params_infer() + self.get_params()

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

            zs, i_costs = self.unpack_infer(z0, outs)

        elif n_inference_steps == 1:
            inps = [ys[0]] + outputs_info[:-1] + non_seqs
            outs = self.step_infer(*inps)
            zs, i_costs = self.unpack_infer(z0, outs)

        elif n_inference_steps == 0:
            zs, i_costs = self.unpack_infer(z0, None)

        return (zs, i_costs), updates

    # Inference
    def inference(self, x, y, n_inference_steps=20, n_samples=100, pass_gradients=False):
        (zs, _), updates = self.infer_q(x, y, n_inference_steps)
        z = zs[-1]

        (prior_energy, h_energy, y_energy, entropy), m_constants = self.m_step(
            x, y, z, n_samples=n_samples)

        constants = [z, entropy] + m_constants

        return (z, prior_energy, h_energy, y_energy, entropy), updates, constants

    def __call__(self, x, y, n_samples=100, n_inference_steps=0,
                 calculate_log_marginal=False, stride=0):
        outs = OrderedDict()
        updates = theano.OrderedUpdates()
        prior = T.nnet.sigmoid(self.z)

        (zs, i_costs), updates_i = self.infer_q(x, y, n_inference_steps)
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
            z = zs[i-1]
            q = T.nnet.sigmoid(z)
            h = self.posterior.sample(
                q, size=(n_samples, q.shape[0], q.shape[1]))

            py = self.conditional(h)

            pys.append(py)
            cond_term = self.conditional.neg_log_prob(y[None, :, :], py).mean()
            kl_term = self.kl_divergence(q, prior[None, :]).mean()
            lower_bounds.append(cond_term + kl_term)

            if calculate_log_marginal:
                nll = -self.log_marginal(y[None, :, :], h, py, q[None, :, :], prior[None, None, :])
                nlls.append(nll)

        outs.update(
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

