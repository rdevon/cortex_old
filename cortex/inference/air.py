'''
Adaptive importance sampling inference.
'''

from collections import OrderedDict
import theano
from theano import tensor as T

from .irvi import IRVI, DeepIRVI
from ..utils import floatX
from ..utils.tools import (
    get_w_tilde,
    scan,
    warn_kwargs
)


class AIR(IRVI):
    '''
    Adaptive importance refinement (AIR).
    
    Inference procedure to refine the posterior using adaptive importance
    sampling (AIS)
    '''
    def __init__(self,
                 model,
                 name='AIR',
                 pass_gradients=False,
                 **kwargs):
        '''Init function for AIR
        
        Args:
            model: Helmholtz object.
            name: str
            pass_gradients: bool (optional)
            kwargs: dict, remaining IRVI arguments.
        '''

        super(AIR, self).__init__(model, name=name,
                                  pass_gradients=pass_gradients,
                                  **kwargs)

    def step_infer(self, r, q, y, *params):
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
        
        model = self.model
        prior_params = model.get_prior_params(*params)

        h        = (r <= q[None, :, :]).astype(floatX)
        py       = model.p_y_given_h(h, *params)
        log_py_h = -model.conditional.neg_log_prob(y[None, :, :], py)
        log_ph   = -model.prior.step_neg_log_prob(h, *prior_params)
        log_qh   = -model.posterior.neg_log_prob(h, q[None, :, :])
        log_p     = log_py_h + log_ph - log_qh
        w_tilde = get_w_tilde(log_p)
        cost    = -log_p.mean()
        q_ = (w_tilde[:, :, None] * h).sum(axis=0)
        q  = self.inference_rate * q_ + (1 - self.inference_rate) * q
        return q, cost

    def init_infer(self, q):
        return []

    def unpack_infer(self, outs):
        return outs

    def params_infer(self):
        return []


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
