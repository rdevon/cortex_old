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
from distributions import Binomial, Gaussian
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
    update_dict_of_lists,
    _slice
)


def unpack(dim_in=None,
           dim_h=None,
           z_init=None,
           recognition_net=None,
           generation_net=None,
           extra_args=dict(),
           distributions=dict(),
           dims=dict(),
           dataset_args=dict(),
           center_input=None,
           **model_args):
    '''
    Function to unpack pretrained model into fresh SFFN class.
    '''

    print 'Unpacking model with parameters %s' % model_args.keys()

    print 'Forming prior model'
    prior_model = Binomial(dim_h)
    models = []

    kwargs = SBN.mlp_factory(dim_h, dims, distributions,
                             recognition_net=recognition_net,
                             generation_net=generation_net)

    models.append(prior_model)
    kwargs['prior'] = prior_model    
    print 'Forming SBN'
    model = SBN(dim_in, dim_h, **kwargs)
    models.append(model)
    models += [model.posterior, model.conditional]

    return models, model_args, extra_args


class SBN(Layer):
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

        kwargs = init_weights(self, **kwargs)
        kwargs = init_rngs(self, **kwargs)

        super(SBN, self).__init__(name=name)

    @staticmethod
    def mlp_factory(dim_h, dims, distributions,
                    recognition_net=None, generation_net=None):
        mlps = {}

        if recognition_net is not None:
            t = recognition_net.get('type', None)
            if t is None:
                input_name = recognition_net.get('input_layer')
                recognition_net['distribution'] = 'binomial'
                recognition_net['dim_in'] = dims[input_name]
                recognition_net['dim_out'] = dim_h
                posterior = MLP.factory(**recognition_net)
            elif t == 'darn':
                input_name = recognition_net.get('input_layer')
                recognition_net['distribution'] = 'binomial'
                recognition_net['dim_in'] = dims[input_name]
                recognition_net['dim_out'] = dim_h
                posterior = DARN.factory(**recognition_net)
            else:
                raise ValueError(t)
            mlps['posterior'] = posterior

        if generation_net is not None:
            output_name = generation_net['output']
            generation_net['dim_in'] = dim_h

            t = generation_net.get('type', None)
            if t is None:
                generation_net['dim_out'] = dims[output_name]
                generation_net['distribution'] = distributions[output_name]
                conditional = MLP.factory(**generation_net)
            elif t == 'darn':
                generation_net['dim_out'] = dims[output_name]
                generation_net['distribution'] = distributions[output_name]
                conditional = DARN.factory(**generation_net)
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
            self.prior = Binomial(self.dim_h)
        elif isinstance(self.prior, Gaussian):
            raise NotImplementedError('Gaussian prior not supported here ATM. '
                                      'Try gbn.py')

        if self.posterior is None:
            self.posterior = MLP(self.dim_in, self.dim_h,
                                 dim_hs=[],
                                 rng=self.rng, trng=self.trng,
                                 h_act='T.nnet.sigmoid',
                                 distribution='binomial')
        elif isinstance(self.posterior, DARN):
            raise ValueError('DARN posterior not supported ATM')

        if self.conditional is None:
            self.conditional = MLP(self.dim_h, self.dim_in,
                                   dim_hs=[],
                                   rng=self.rng, trng=self.trng,
                                   h_act='T.nnet.sigmoid',
                                   distribution='binomial')

        self.posterior.name = self.name + '_posterior'
        self.conditional.name = self.name + '_conditional'

    def set_tparams(self, excludes=[]):
        excludes = ['{name}_{key}'.format(name=self.name, key=key)
                    for key in excludes]
        tparams = super(SBN, self).set_tparams()
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

    # E ---------------------------------------------------------

    def sample_from_prior(self, n_samples=99):
        h, updates = self.prior.sample(n_samples)
        return self.conditional.feed(h), updates

    def generate_from_latent(self, h):
        py = self.conditional.feed(h)
        center = self.conditional.get_center(py)
        return center

    def visualize_latents(self):
        h = T.eye(self.prior.dim).astype(floatX)
        py = self.conditional.get_center(self.conditional.feed(h))
        h0 = T.zeros_like(h)
        py0 = self.conditional.get_center(self.conditional.feed(h0))
        return py - py0

    # Misc --------------------------------------------------------------------

    def get_center(self, p):
        return self.conditional.get_center(p)

    def log_marginal(self, y, h, py, q):
        log_py_h = -self.conditional.neg_log_prob(y, py)
        log_ph   = -self.prior.neg_log_prob(h)
        log_qh   = -self.posterior.neg_log_prob(h, q)
        assert log_py_h.ndim == log_ph.ndim == log_qh.ndim

        log_p     = log_py_h + log_ph - log_qh
        log_p_max = T.max(log_p, axis=0, keepdims=True)
        w         = T.exp(log_p - log_p_max)

        return (T.log(w.mean(axis=0, keepdims=True)) + log_p_max).mean()

    def l2_decay(self, rate):
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
        start  = self.prior.n_params
        stop   = start + self.conditional.n_params
        params = params[start:stop]
        return self.conditional.step_feed(h, *params)

    def __call__(self, x, y, qk=None, n_posterior_samples=10):
        q0  = self.posterior.feed(x)

        if qk is None:
            qk = q0

        r   = self.prior.prototype_samples(
            (n_posterior_samples, y.shape[0], self.dim_h))
        h   = (r <= qk[None, :, :]).astype(floatX)
        py  = self.conditional.feed(h)

        log_ph   = -self.prior.neg_log_prob(h)
        log_qh   = -self.posterior.neg_log_prob(h, q0[None, :, :])
        log_qkh  = -self.posterior.neg_log_prob(h, qk[None, :, :])
        log_py_h = -self.conditional.neg_log_prob(y[None, :, :], py)

        log_p         = log_sum_exp(log_py_h + log_ph - log_qkh, axis=0) - T.log(n_posterior_samples)

        y_energy      = -log_py_h.mean(axis=0)
        prior_energy  = -log_ph.mean(axis=0)
        h_energy      = -log_qh.mean(axis=0)

        nll           = -log_p
        prior_entropy = self.prior.entropy()
        q_entropy     = self.posterior.entropy(qk)

        cost = (y_energy + prior_energy + h_energy).sum(0)
        lower_bound = -(y_energy + prior_energy - q_entropy).mean()

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
            py=py,
            batch_energies=y_energy
        )

        return results, samples, theano.OrderedUpdates()
