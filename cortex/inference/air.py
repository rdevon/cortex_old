'''
Adaptive importance sampling inference.
'''

from collections import OrderedDict
import theano
from theano import tensor as T

from .irvi import IRVI, DeepIRVI
from ..utils import floatX, scan
from ..utils.maths import norm_exp


class AIR(IRVI):
    '''
    Adaptive importance refinement (AIR).

    Inference procedure to refine the posterior using adaptive importance
    sampling (AIS)
    '''
    _required = ['prior', 'conditional', 'posterior']
    _components = {
        'prior': {},
        'conditional': {},
        'posterior': {}}

    def __init__(self, prior, conditional, posterior, name='AIR', **kwargs):
        '''Init function for AIR

        Args:
            name: str
            kwargs: dict, remaining IRVI arguments.
        '''

        models = dict(
            prior=prior,
            conditional=conditional,
            posterior=posterior)

        super(AIR, self).__init__(name=name, models=models, **kwargs)

    def step_infer(self, r, q, y, inference_rate, *params):
        '''Step inference function for IRVI.inference scan.

        Args:
            r: theano randomstream variable
            q: T.tensor. Current approximate posterior parameters
            y: T.tensor. Data sample
            params: list of shared variables
        Returns:
            q: T.tensor. New approximate posterior parameters
            cost: T.scalar float. Negative lower bound of current parameters
        '''

        h = self.posterior.quantile(r, q)
        weights, cost = self.score(h, y, q, *params)

        q_ = (weights[:, :, None] * h).sum(axis=0)
        q  = inference_rate * q_ + (1 - inference_rate) * q
        return q, cost

    def score(self, h, y, q, *params):
        prior_params = self.select_params('prior', *params)
        post_params = self.select_params('posterior', *params)
        cond_params = self.select_params('conditional', *params)
        py       = self.conditional._feed(h, *cond_params)['P']
        log_py_h = -self.conditional.neg_log_prob(y[None, :, :], P=py)
        log_ph   = -self.prior.step_neg_log_prob(h, *prior_params)
        log_qh   = -self.posterior.neg_log_prob(h, P=q[None, :, :])
        log_p     = log_py_h + log_ph - log_qh
        w_tilde = norm_exp(log_p)
        cost    = -log_p.mean()
        return w_tilde, -log_p.mean()

    def init_infer(self, q):
        return []

    def unpack_infer(self, outs):
        return outs

    def params_infer(self, inference_rate=None):
        return [inference_rate]

    def generate_random_variables(self, shape, P=None):
        if P is None:
            session = self.manager.get_session()
            P = session.tensors[self.name + 'Qk']

        return self.posterior.generate_random_variables(shape, P=P)

    def _sample(self, epsilon, P=None):
        session = self.manager.get_session()
        if P is None:
            if _p(self.name, 'Qk') not in session.tensors.keys():
                raise TypeError('%s.Qk not found in graph nor provided'
                                % self.name)
            P = session.tensors[_p(self.name, 'Qk')]
        return self.posterior._sample(epsilon, P=P)


class DeepAIR(DeepIRVI):
    def __init__(self,
                 model,
                 name='AIR',
                 pass_gradients=False,
                 **kwargs):

        super(DeepAIR, self).__init__(model, name=name,
                                      pass_gradients=pass_gradients,
                                      **kwargs)

    def step_infer(self, *params):
        model = self.model

        params       = list(params)
        rs           = params[:model.n_layers]
        qs           = params[model.n_layers:2*model.n_layers]
        y            = params[2*model.n_layers]
        params       = params[1+2*model.n_layers:]
        prior_params = model.get_prior_params(*params)

        hs     = []
        new_qs = []

        for l, (q, r) in enumerate(zip(qs, rs)):
            h = (r <= q[None, :, :]).astype(floatX)
            hs.append(h)

        ys   = [y[None, :, :]] + hs[:-1]
        p_ys = [model.p_y_given_h(h, l, *params) for l, h in enumerate(hs)]

        log_ph = -model.prior.step_neg_log_prob(hs[-1], *prior_params)
        log_py_h = T.constant(0.).astype(floatX)
        log_qh = T.constant(0.).astype(floatX)
        for l in xrange(model.n_layers):
            log_py_h += -model.conditionals[l].neg_log_prob(ys[l], p_ys[l])
            log_qh += -model.posteriors[l].neg_log_prob(hs[l], qs[l][None, :, :])

        log_p   = log_py_h + log_ph - log_qh
        w_tilde = get_w_tilde(log_p)
        cost = -log_p.mean()

        for q, h in zip(qs, hs):
            q_ = (w_tilde[:, :, None] * h).sum(axis=0)
            new_qs.append(self.inference_rate * q_ + (1 - self.inference_rate) * q)

        return tuple(new_qs) + (cost,)

    def init_infer(self, qs):
        return []

    def unpack_infer(self, outs):
        return outs[:-1], outs[-1]

    def params_infer(self):
        return []

_classes = {'AIR': AIR}