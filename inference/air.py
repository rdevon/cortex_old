'''
Adaptive importance sampling inference.
'''

from collections import OrderedDict
import theano
from theano import tensor as T

from irvi import IRVI
from utils import floatX
from utils.tools import (
    scan,
    warn_kwargs
)


class AIR(IRVI):
    def __init__(self,
                 model,
                 name='AIR',
                 **kwargs):

        super(AIR, self).__init__(model, name=name,
                                  pass_gradients=False, **kwargs)

    def step_infer(self, r, q, y, *params):
        model = self.model
        prior_params = model.get_prior_params(*params)

        h        = (r <= q[None, :, :]).astype(floatX)
        py       = model.p_y_given_h(h, *params)
        log_py_h = -model.conditional.neg_log_prob(y[None, :, :], py)
        log_ph   = -model.prior.step_neg_log_prob(h, model.prior.get_prob(*prior_params))
        log_qh   = -model.posterior.neg_log_prob(h, q[None, :, :])
    
        log_p     = log_py_h + log_ph - log_qh
        log_p_max = T.max(log_p, axis=0, keepdims=True)

        w       = T.exp(log_p - log_p_max)
        w_tilde = w / w.sum(axis=0, keepdims=True)
        cost    = log_p.mean()
        q_ = (w_tilde[:, :, None] * h).sum(axis=0)
        q  = self.inference_rate * q_ + (1 - self.inference_rate) * q
        return q, cost

    def init_infer(self, q):
        return []

    def unpack_infer(self, outs):
        return outs

    def params_infer(self):
        return []
