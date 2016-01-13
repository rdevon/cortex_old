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


class RNN_SBN(Layer):
    def __init__(self, dim_in, dim_h,
                 posterior=None, conditional=None, aux_net=None,
                 name='rnn_sbn',
                 **kwargs):

        self.dim_in = dim_in
        self.dim_h = dim_h

        self.posterior = posterior
        self.conditional = conditional
        self.aux_net = aux_net

        kwargs = init_inference_args(self, **kwargs)
        kwargs = init_weights(self, **kwargs)
        kwargs = init_rngs(self, **kwargs)

        super(RNN_SBN, self).__init__(name=name)

    @staticmethod
    def rnn_factory(recognition_rnn=None, generation_rnn=None):
        rnns = {}

        if recognition_rnn is not None:
            t = recognition_rnn.get('type', None)
            if t == 'RNN' or t is None:
                C = RNN
            elif t == 'GRU':
                C = GRU
            else:
                raise ValueError(t)
            r_dict = OrderedDict((k, v) for (k, v) in recognition_rnn.iteritems())
            nets = r_dict.pop('nets')

            if nets is not None:
                mlps = C.mlp_factory(r_dict['dim_in'],
                                     r_dict['dim_h'],
                                     dim_out=r_dict['dim_out'],
                                     **nets)
                r_dict.update(**mlps)

            posterior = C.factory(**r_dict)
            rnns['posterior'] = posterior

        if generation_rnn is not None:
            t = generation_rnn.get('type', None)
            if t == 'RNN' or t is None:
                C = RNN
            elif t == 'GRU':
                C = GRU
            else:
                raise ValueError(t)
            g_dict = OrderedDict((k, v) for (k, v) in generation_rnn.iteritems())
            nets = g_dict.pop('nets')

            if nets is not None:
                mlps = C.mlp_factory(g_dict['dim_in'],
                                     g_dict['dim_h'],
                                     dim_out=g_dict['dim_out'],
                                     **nets)
                g_dict.update(**mlps)

            conditional = C.factory(**g_dict)
            rnns['conditional'] = conditional

        return rnns

    def set_params(self):
        z = np.zeros((self.dim_h,)).astype(floatX)
        inference_scale_factor = np.float32(1.0)

        self.params = OrderedDict(z=z, inference_scale_factor=inference_scale_factor)

        if self.posterior is None:
            self.posterior = RNN(self.dim_in, self.dim_h)
        if self.conditional is None:
            self.conditional = RNN(self.dim_h, self.dim_in)
        if self.aux_net is None:
            self.aux_net = MLP.factory(dim_in=self.dim_h, dim_h=0,
                                       dim_out=self.conditional.dim_h,
                                       n_layers=1, out_act='T.tanh')

    def set_tparams(self, excludes=[]):
        excludes.append('inference_scale_factor')
        excludes = ['{name}_{key}'.format(name=self.name, key=key)
                    for key in excludes]
        tparams = super(RNN_SBN, self).set_tparams()
        tparams.update(**self.posterior.set_tparams())
        tparams.update(**self.conditional.set_tparams())
        tparams.update(**self.aux_net.set_tparams())
        tparams = OrderedDict((k, v) for k, v in tparams.iteritems()
            if k not in excludes)

        return tparams

    def get_params(self):
        params = [self.z] + self.conditional.get_sample_params() + self.posterior.get_net_params() + self.aux_net.get_params() + [self.inference_scale_factor]
        return params

    def p_y_given_h(self, preact, h, *params):
        params = params[1:1+len(self.conditional.get_params())]
        return self.conditional.step_call(h, *params)

    def sample_from_prior(self, n_samples=100):
        p = T.nnet.sigmoid(self.z)
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
        y_energy = self.conditional.neg_log_prob(y, py) # is implemented
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
        entropy_term = self.posterior.output_net.entropy(p)
        prior_term = self.posterior.neg_log_prob(p, q)
        return prior_term - entropy_term

    def m_step(self, x, y, z, p_h, n_samples=10, assign=False, sample=False):
        constants = []
        q = T.nnet.sigmoid(z)
        prior = T.nnet.sigmoid(self.z)

        h = self.posterior.output_net.sample(
            q, size=(n_samples, q.shape[0], q.shape[1]))

        cond = self.conditional.conditional(h, return_preact=True)
        updates = theano.OrderedUpdates()

        if assign:
            print 'Assignment at M step'
            y_e = T.zeros((y.shape[0], cond.shape[0], cond.shape[1], y.shape[2])).astype(floatX) + y[:, None, :, :]
            cond_h0 = self.aux_net(h)

            y_e = y_e.reshape((y.shape[0], cond.shape[0] * cond.shape[1], y.shape[2])).astype(floatX)
            cond_e = cond.reshape((cond.shape[0] * cond.shape[1], cond.shape[2])).astype(floatX)
            cond_h0 = cond_h0.reshape((cond.shape[0] * cond.shape[1], cond_h0.shape[2])).astype(floatX)

            out_dict, updates = self.conditional.assign(y_e, h0=cond_h0, condition_on=cond_e, select_first=True)
            py = out_dict['probs']
            py = py.reshape((py.shape[0], cond.shape[0], cond.shape[1], py.shape[2])).astype(floatX)
            cond_dict, _ = self.conditional(y_e[:-1], h0=cond_h0, condition_on=cond_e)

            y_energy = out_dict['energies'].sum(axis=0).mean()
        else:
            print 'Regular M step'
            y_e = T.zeros((y.shape[0], cond.shape[0], cond.shape[1], y.shape[2])).astype(floatX) + y[:, None, :, :]
            cond_h0 = self.aux_net(h)

            y_e = y_e.reshape((y.shape[0], cond.shape[0] * cond.shape[1], y.shape[2])).astype(floatX)
            cond_e = cond.reshape((cond.shape[0] * cond.shape[1], cond.shape[2])).astype(floatX)
            cond_h0 = cond_h0.reshape((cond.shape[0] * cond.shape[1], cond_h0.shape[2])).astype(floatX)

            cond_dict, _ = self.conditional(y_e[:-1], h0=cond_h0, condition_on=cond_e)
            py = cond_dict['p']
            py = py.reshape((py.shape[0], cond.shape[0], cond.shape[1], py.shape[2])).astype(floatX)

            y_energy = self.conditional.neg_log_prob(y[:, None, :, :][1:], py).sum(axis=0).mean()

        entropy = self.posterior.output_net.entropy(q).mean()
        prior_energy = self.posterior.neg_log_prob(q, prior[None, :]).mean()
        h_energy = self.posterior.neg_log_prob(q, p_h).mean()
        constants.append(entropy)

        '''
        print 'Getting conditional h0 cost'
        z_e = concatenate([y_e, h.reshape((h.shape[0] * h.shape[1], h.shape[2]))[None, :, :]], axis=2)
        cond_h0s = self.aux_net(z_e)
        cond_h_c = cond_dict['h'].copy()
        constants.append(cond_h_c)
        cond_h_cost = ((cond_h0s[1:] - cond_h_c) ** 2).sum(axis=2).mean()
        '''

        return OrderedDict(prior_energy=prior_energy, h_energy=h_energy,
                           y_energy=y_energy, entropy=entropy), constants, updates

    def step_infer(self, *params):
        raise NotImplementedError()

    def init_infer(self, z):
        raise NotImplementedError()

    def unpack_infer(self, outs):
        raise NotImplementedError()

    def params_infer(self):
        raise NotImplementedError()

    def _step_momentum(self): pass
    def _init_momentum(self): pass
    def _unpack_momentum(self): pass
    def _params_momentum(self): pass

    # Importance Sampling
    def _step_adapt_simple(self, q, y, preact, *params):
        params = list(params)
        prior = T.nnet.sigmoid(params[0])
        h = self.posterior.output_net.sample(
            q, size=(self.n_inference_samples, q.shape[0], q.shape[1]))

        cond_params = params[1:1+len(self.conditional.get_sample_params())]
        c_params = self.conditional.get_conditional_args(*cond_params)

        cond = self.conditional.conditional.step_call(h, *c_params)
        py = eval(self.conditional.output_net.out_act)(preact[:, None, :, :] + cond[None, :, :, :])

        y_energy = self.conditional.neg_log_prob(y[:, None, :, :], py).sum(axis=0)
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

    def _step_adapt(self, q, y, *params):
        print 'Adaptive inference'
        params = list(params)
        prior = T.nnet.sigmoid(params[0])
        h = self.posterior.output_net.sample(
            q, size=(self.n_inference_samples, q.shape[0], q.shape[1]))

        cond_params = params[1:1+len(self.conditional.get_sample_params())]
        c_params = self.conditional.get_conditional_args(*cond_params)
        cond = self.conditional.conditional.preact(h, *c_params)

        aux_params = params[-(1+len(self.aux_net.get_params())):-1]
        y_e = T.zeros((y.shape[0], cond.shape[0], cond.shape[1], y.shape[2])).astype(floatX) + y[:, None, :, :]
        cond_h0 = self.aux_net.step_call(h, *aux_params)

        y_e = y_e.reshape((y.shape[0], cond.shape[0] * cond.shape[1], y.shape[2])).astype(floatX)
        cond_e = cond.reshape((cond.shape[0] * cond.shape[1], cond.shape[2])).astype(floatX)
        cond_h0 = cond_h0.reshape((cond.shape[0] * cond.shape[1], cond_h0.shape[2])).astype(floatX)

        cond_dict, _ = self.conditional.step_call(y_e[:-1], cond_h0, cond_e, *cond_params)
        py = cond_dict['p']
        py = py.reshape((py.shape[0], cond.shape[0], cond.shape[1], py.shape[2])).astype(floatX)

        y_energy = self.conditional.neg_log_prob(y[:, None, :, :][1:], py).sum(axis=0)
        prior_energy = self.posterior.neg_log_prob(h, prior[None, None, :])
        entropy_term = self.posterior.neg_log_prob(h, q[None, :, :])

        log_p = -y_energy - prior_energy + entropy_term
        log_p_max = T.max(log_p, axis=0, keepdims=True)
        w = T.exp(log_p - log_p_max)

        w_tilde = w / w.sum(axis=0, keepdims=True)

        cost = (log_p - log_p_max).mean()
        q_ = (w_tilde[:, :, None] * h).sum(axis=0)
        q = self.inference_rate * q_ + (1 - self.inference_rate) * q

        return q, prior_energy.mean()

    def _step_adapt_assign(self, q, y, *params):
        print 'Assignment adapt inference'
        params = list(params)
        prior = T.nnet.sigmoid(params[0])
        h = self.posterior.output_net.sample(
            q, size=(self.n_inference_samples, q.shape[0], q.shape[1]))

        cond_params = params[1:1+len(self.conditional.get_sample_params())]
        c_params = self.conditional.get_conditional_args(*cond_params)
        cond = self.conditional.conditional.preact(h, *c_params)

        aux_params = params[-(1+len(self.aux_net.get_params())):-1]
        y_e = T.zeros((y.shape[0], cond.shape[0], cond.shape[1], y.shape[2])).astype(floatX) + y[:, None, :, :]
        cond_h0 = self.aux_net.step_call(h, *aux_params)

        y_e = y_e.reshape((y.shape[0], cond.shape[0] * cond.shape[1], y.shape[2])).astype(floatX)
        cond_e = cond.reshape((cond.shape[0] * cond.shape[1], cond.shape[2])).astype(floatX)
        cond_h0 = cond_h0.reshape((cond.shape[0] * cond.shape[1], cond_h0.shape[2])).astype(floatX)

        cond_dict, _ = self.conditional.step_assign_call(y_e, cond_h0, cond_e,
                                                         0.0, 1.0, y.shape[0] - 1,
                                                         False, True, *cond_params)
        py = cond_dict['probs']
        py = py.reshape((py.shape[0], cond.shape[0], cond.shape[1], py.shape[2])).astype(floatX)

        y_energy = self.conditional.neg_log_prob(y[:, None, :, :], py).sum(axis=0)
        prior_energy = self.posterior.neg_log_prob(h, prior[None, None, :])
        entropy_term = self.posterior.neg_log_prob(h, q[None, :, :])

        log_p = -y_energy - prior_energy + entropy_term
        log_p_max = T.max(log_p, axis=0, keepdims=True)
        w = T.exp(log_p - log_p_max)

        w_tilde = w / w.sum(axis=0, keepdims=True)

        cost = (log_p - log_p_max).mean()
        q_ = (w_tilde[:, :, None] * h).sum(axis=0)
        q = self.inference_rate * q_ + (1 - self.inference_rate) * q

        return q, prior_energy.mean()

    def _init_adapt(self, q):
        return []

    def _init_variational_params_adapt(self, p_h_logit):
        print 'Starting z0 at recognition net'
        q0 = T.nnet.sigmoid(p_h_logit)

        return q0

    def _unpack_adapt(self, q0, outs):
        if outs is not None:
            qs, costs = outs
            if qs.ndim == 2:
                qs = concatenate([q0[None, :, :], qs[None, :, :]], axis=0)
                costs = [costs]
            else:
                qs = T.concatenate([q0[None, :, :], qs])

        else:
            qs = q0[None, :, :]
            costs = [T.constant(0.).astype(floatX)]
        return logit(qs), costs

    def _params_adapt(self):
        return []

    def assign(self, y, z, apply_assignment=False):
        def step_order(y, idx):
            return y[idx.astype('int64')]

        cond = self.conditional.conditional(z, return_preact=True)
        y_e = T.zeros((y.shape[0], cond.shape[0], cond.shape[1], y.shape[2])).astype(floatX) + y[:, None, :, :]
        cond_h0 = self.aux_net(z)

        y_e = y_e.reshape((y.shape[0], cond.shape[0] * cond.shape[1], y.shape[2])).astype(floatX)
        cond_e = cond.reshape((cond.shape[0] * cond.shape[1], cond.shape[2])).astype(floatX)
        cond_h0 = cond_h0.reshape((cond.shape[0] * cond.shape[1], cond_h0.shape[2])).astype(floatX)

        out_dict, updates = self.conditional.assign(y_e, h0=cond_h0, condition_on=cond_e, select_first=True)

        if apply_assignment:
            y_hat, updates_a = scan(step_order, [y_e.transpose(1, 0, 2), out_dict['chain'].T], [None], [],
                                    y_e.shape[1], name='order_data', strict=False)
            updates.update(updates_a)
            y_hat = y_hat.transpose(1, 0, 2)
            y_hat = y_hat.reshape((y_hat.shape[0], cond.shape[0], cond.shape[1], y_hat.shape[2])).astype(floatX)
            out_dict['y_hat'] = y_hat

        out_dict['h0'] = cond_h0

        return out_dict, updates

    def infer_q(self, x, y, p_h_logit, n_inference_steps):
        updates = theano.OrderedUpdates()

        z0 = self.init_variational_params(p_h_logit)

        seqs = []
        outputs_info = [z0] + self.init_infer(z0) + [None]
        #non_seqs = [y, cond_preact] + self.params_infer() + self.get_params()
        non_seqs = [y] + self.params_infer() + self.get_params()

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
            inps = outputs_info[:-1] + non_seqs
            outs = self.step_infer(*inps)
            zs, i_costs = self.unpack_infer(z0, outs)

        elif n_inference_steps == 0:
            zs, i_costs = self.unpack_infer(z0, None)

        return (zs, i_costs), updates

    # Inference
    def inference(self, x, y, post_h0=None, n_inference_steps=20,
                  n_samples=100, pass_gradients=False, sample=False, assign=False):
        #cond_dict, updates = self.conditional(y, h0=cond_h0)
        #cond_preact = cond_dict['z']

        post_dict, updates = self.posterior(x, h0=post_h0)
        p_h = post_dict['p'][-1]
        p_h_logit = post_dict['z'][-1]

        #(zs, _), updates_i = self.infer_q(x, y, p_h_logit, cond_preact, n_inference_steps)
        (zs, _), updates_i = self.infer_q(x, y, p_h_logit, n_inference_steps)
        z = zs[-1]

        updates.update(updates_i)

        out_dict, m_constants, updates_m = self.m_step(
            x, y, z, p_h, n_samples=n_samples, sample=sample, assign=assign)
        updates.update(updates_m)
        out_dict.update(**post_dict)

        constants = [z] + m_constants

        return out_dict, updates, constants

    def __call__(self, x, y, post_h0=None, n_samples=100, n_inference_steps=0,
                 calculate_log_marginal=False, stride=0, assign=False):
        outs = OrderedDict()
        updates = theano.OrderedUpdates()
        prior = T.nnet.sigmoid(self.z)

        post_dict, updates = self.posterior(x, h0=post_h0)
        p_h = post_dict['p'][-1]
        p_h_logit = post_dict['z'][-1]

        (zs, i_costs), updates_i = self.infer_q(x, y, p_h_logit, n_inference_steps)
        updates.update(updates_i)

        if n_inference_steps > stride and stride != 0:
            steps = [0] + range(n_inference_steps // 10, n_inference_steps + 1, n_inference_steps // 10)
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

            h = self.posterior.output_net.sample(
                q, size=(n_samples, q.shape[0], q.shape[1]))

            if assign:
                out_dict, updates_a = self.assign(y, h, apply_assignment=True)
                updates.update(**updates_a)
                cond_term = out_dict['energies'].sum(axis=0).mean()
                py = out_dict['probs']
                py = py.reshape((py.shape[0], h.shape[0], h.shape[1], py.shape[2])).astype(floatX)[1:]
                y_hat = out_dict['y_hat']
                h0 = out_dict['h0']
                #y_hat_e = y_hat.reshape((y_hat.shape[0], h0.shape[1], y_hat.shape[2])).astype(floatX)
                #s_dict, updates_sam = self.conditional.sample(
                #    x0=y_hat_e[0], h0=h0, n_steps=y_hat_e.shape[0]-1,
                #    condition_on=h.reshape((h.shape[0] * h.shape[1], h.shape[2])))
                #py = s_dict['p'][1:]
                #py = py.reshape((py.shape[0], h.shape[0], h.shape[1], py.shape[2]))
                #cond_term = self.conditional.neg_log_prob(out_dict['y_hat'], s_dict['p']).sum(axis=0).mean()
                #updates += updates_sam
                outs.update(y_hat=y_hat)
            else:
                cond = self.conditional.conditional(h, return_preact=True)
                y_e = T.zeros((y.shape[0], cond.shape[0], cond.shape[1], y.shape[2])).astype(floatX) + y[:, None, :, :]
                cond_h0 = self.aux_net(h)

                y_e = y_e.reshape((y.shape[0], cond.shape[0] * cond.shape[1], y.shape[2])).astype(floatX)
                cond_e = cond.reshape((cond.shape[0] * cond.shape[1], cond.shape[2])).astype(floatX)
                cond_h0 = cond_h0.reshape((cond.shape[0] * cond.shape[1], cond_h0.shape[2])).astype(floatX)

                cond_dict, _ = self.conditional(y_e[:-1], cond_h0, cond_e)
                py = cond_dict['p']
                py = py.reshape((py.shape[0], cond.shape[0], cond.shape[1], py.shape[2])).astype(floatX)
                cond_term = self.conditional.neg_log_prob(y[:, None, :, :][1:], py).sum(axis=0).mean()

            pys.append(py)

            kl_term = self.kl_divergence(q, prior[None, :]).mean()
            lower_bounds.append(cond_term + kl_term)

            if calculate_log_marginal:
                nll = -self.log_marginal(y[:, None, :, :][1:], h, py, q[None, :, :], prior[None, None, :])
                nlls.append(nll)

        outs.update(
            zs=zs,
            py0=pys[0],
            py=pys[-1],
            lower_bound=lower_bounds[-1],
            lower_bounds=lower_bounds,
            inference_cost=(lower_bounds[0] - lower_bounds[-1])
        )

        if calculate_log_marginal:
            outs.update(nll=nlls[-1], nlls=nlls)

        return outs, updates
