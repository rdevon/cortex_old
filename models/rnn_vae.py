'''
module of Stochastic Feed Forward Networks
'''

from collections import OrderedDict
import numpy as np
import theano
from theano import tensor as T

from layers import Layer
from gru import GRU
from mlp import MLP
from rnn import RNN
from sbn import (
    init_inference_args,
    init_momentum_args,
    init_sgd_args,
    set_input,
    _noise,
    _sample,
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
    scan,
    _slice
)


def unpack(dim_in=None,
           dim_h=None,
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


class RNN_VAE(Layer):
    def __init__(self, dim_in, dim_h,
                 posterior=None, conditional=None, aux_net=None,
                 name='rnn_vae',
                 **kwargs):

        self.dim_in = dim_in
        self.dim_h = dim_h

        self.posterior = posterior
        self.conditional = conditional
        self.aux_net = aux_net

        kwargs = init_weights(self, **kwargs)
        kwargs = init_rngs(self, **kwargs)

        super(RNN_VAE, self).__init__(name=name)

    def set_params(self):
        mu = np.zeros((self.dim_h,)).astype(floatX)
        log_sigma = np.zeros((self.dim_h,)).astype(floatX)

        self.params = OrderedDict(
            mu=mu, log_sigma=log_sigma)

        if self.posterior is None:
            self.posterior = RNN(self.dim_in, self.dim_h)
        if self.conditional is None:
            self.conditional = RNN(self.dim_h, self.dim_in)
        if self.aux_net is None:
            self.aux_net = MLP.factory(dim_in=self.dim_h, dim_h=0,
                                       dim_out=self.conditional.dim_h,
                                       n_layers=1, out_act='T.tanh')

    def set_tparams(self, excludes=[]):
        excludes = [ex for ex in excludes if ex in self.params.keys()]
        print 'Excluding the following parameters from learning: %s' % excludes
        excludes = ['{name}_{key}'.format(name=self.name, key=key)
                    for key in excludes]
        tparams = super(RNN_VAE, self).set_tparams()
        tparams.update(**self.posterior.set_tparams())
        tparams.update(**self.conditional.set_tparams())
        tparams.update(**self.aux_net.set_tparams())
        tparams = OrderedDict((k, v) for k, v in tparams.iteritems()
            if k not in excludes)

        return tparams

    def get_params(self):
        params = [self.mu, self.log_sigma] + self.conditional.get_sample_params() + self.posterior.get_net_params() + self.aux_net.get_params()
        return params

    def p_y_given_h(self, preact, h, *params):
        params = params[2:2+len(self.conditional.get_params())]
        return self.conditional.step_call(h, *params)

    def sample_from_prior(self, n_samples=100):
        p = T.concatenate([self.mu, self.log_sigma])
        h = self.posterior.output_net.sample(p=p, size=(n_samples, self.dim_h))
        return h

    def generate_from_prior(self, n_samples=100):
        h = self.sample_from_prior(n_samples=n_samples)
        cond_dict, updates = self.conditional.sample(condition_on=h)
        py = cond_dict['p']
        return py, updates

    def generate_from_latent(self, H):
        py = self.conditional(H)
        prob = self.conditional.prob(py)
        return prob

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
        y_energy = self.conditional.neg_log_prob(y, py).sum(axis=0)
        prior_energy = self.posterior.neg_log_prob(h, prior)
        entropy_term = self.posterior.neg_log_prob(h, q)

        log_p = -y_energy - prior_energy + entropy_term
        log_p_max = T.max(log_p, axis=0, keepdims=True)
        w = T.exp(log_p - log_p_max)

        return (T.log(w.mean(axis=0, keepdims=True)) + log_p_max).mean()

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

    def assign(self, y, z, apply_assignment=False):
        def step_order(y, idx):
            return y[idx.astype('int64')]

        cond = self.conditional.conditional(z, return_preact=True)
        cond_h0 = self.aux_net(z)
        out_dict, updates = self.conditional.assign(y, h0=cond_h0, condition_on=cond, select_first=True)

        if apply_assignment:
            y_hat, updates_a = scan(step_order, [y.transpose(1, 0, 2), out_dict['chain'].T], [None], [],
                                    y.shape[1], name='order_data', strict=False)
            updates.update(updates_a)
            y_hat = y_hat.transpose(1, 0, 2)
            out_dict['y_hat'] = y_hat

        out_dict['h0'] = cond_h0

        return out_dict, updates

    def m_step(self, x, y, q, n_samples=10, assign=False):
        updates = theano.OrderedUpdates()
        constants = []

        y = T.zeros((y.shape[0], n_samples, y.shape[1], y.shape[2])
            ).astype(floatX) + y[:, None, :, :]
        y_e = y.reshape((y.shape[0], y.shape[1] * y.shape[2], y.shape[3])).astype(floatX)

        prior = T.concatenate([self.mu[None, :], self.log_sigma[None, :]], axis=1)

        h = self.posterior.output_net.sample(
            q, size=(n_samples, q.shape[0], q.shape[1] // 2))
        h_e = h.reshape((h.shape[0] * h.shape[1], h.shape[2])).astype(floatX)

        cond = self.conditional.conditional(h_e, return_preact=True)
        cond_h0 = self.aux_net(h_e)

        if assign:
            print 'Assignment at M step'
            out_dict, updates = self.conditional.assign(y_e, h0=cond_h0, condition_on=cond, select_first=True)
            y_energy = out_dict['energies'].sum(axis=0).mean()
        else:
            print 'Regular M step'
            cond_dict, _ = self.conditional(y_e[:-1], h0=cond_h0, condition_on=cond)
            py = cond_dict['p']
            y_energy = self.conditional.neg_log_prob(y_e[1:], py).sum(axis=0).mean()

        entropy = self.posterior.output_net.entropy(q).mean()
        prior_energy = self.kl_divergence(q, prior).mean()
        h_energy = T.constant(0.).astype(floatX)
        constants.append(entropy)

        return OrderedDict(prior_energy=prior_energy, h_energy=h_energy,
                           y_energy=y_energy, entropy=entropy), constants, updates

    def inference(self, x, y, post_h0=None, n_samples=100, pass_gradients=False,
                  assign=False):

        post_dict, updates = self.posterior(x, h0=post_h0)
        q = post_dict['p'][-1]

        out_dict, m_constants, updates_m = self.m_step(
            x, y, q, n_samples=n_samples, assign=assign)
        updates.update(updates_m)
        out_dict.update(**post_dict)

        constants = m_constants

        return out_dict, updates, constants

    def __call__(self, x, y, post_h0=None, n_samples=100,
                 calculate_log_marginal=False, stride=0, assign=False):
        outs = OrderedDict()
        prior = T.concatenate([self.mu[None, :], self.log_sigma[None, :]], axis=1)

        # INFERENCE ----------------
        post_dict, updates = self.posterior(x, h0=post_h0)
        q = post_dict['p'][-1]

        # EVAL ---------------------
        y = T.zeros((y.shape[0], n_samples, y.shape[1], y.shape[2])
            ).astype(floatX) + y[:, None, :, :]
        y_e = y.reshape((y.shape[0], y.shape[1] * y.shape[2], y.shape[3])).astype(floatX)

        h = self.posterior.output_net.sample(
            q, size=(n_samples, q.shape[0], q.shape[1] // 2))
        h_e = h.reshape((h.shape[0] * h.shape[1], h.shape[2])).astype(floatX)

        if assign:
            out_dict, updates_a = self.assign(y_e, h_e, apply_assignment=True)
            updates.update(**updates_a)
            cond_term = out_dict['energies'].sum(axis=0).mean()
            py = out_dict['probs']
            py = py.reshape((py.shape[0], y.shape[1], y.shape[2], py.shape[2])).astype(floatX)[1:]
            y_hat = out_dict['y_hat']
            y_hat = y_hat.reshape(
                (y_hat.shape[0], y.shape[1], y.shape[2], y_hat.shape[2])).astype(floatX)
            outs.update(y_hat=y_hat)
        else:
            cond = self.conditional.conditional(h_e, return_preact=True)
            cond_h0 = self.aux_net(h_e)
            cond_dict, _ = self.conditional(y_e[:-1], cond_h0, cond)
            py = cond_dict['p']
            py = py.reshape((py.shape[0], y.shape[1], y.shape[2], py.shape[2])).astype(floatX)
            cond_term = self.conditional.neg_log_prob(y[1:], py).sum(axis=0).mean()

        kl_term = self.kl_divergence(q, prior).mean()
        lower_bound = cond_term + kl_term

        if calculate_log_marginal:
            nll = -self.log_marginal(y[1:], h, py, q[None, :, :], prior[None, :, :])

        outs.update(
            py=py,
            lower_bound=lower_bound
        )

        if calculate_log_marginal:
            outs.update(nll=nll)

        return outs, updates
