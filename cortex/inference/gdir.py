'''
Gradient-Descent Iterative Refinement
'''

from collections import OrderedDict
import theano
from theano import tensor as T

from .irvi import IRVI
from ..utils import floatX
from ..utils.tools import (
    scan,
    update_dict_of_lists,
)


class GDIR(IRVI):
    def __init__(self,
                 model,
                 name='GDIR',
                 pass_gradients=True,
                 **kwargs):

        super(GDIR, self).__init__(model, name=name,
                                   pass_gradients=pass_gradients,
                                   **kwargs)

    def e_step(self, epsilon, q, y, *params):
        model        = self.model
        prior_params = model.get_prior_params(*params)
        h            = model.prior.step_sample(epsilon, q)
        py           = model.p_y_given_h(h, *params)

        consider_constant = [y] + list(params)

        log_py_h = -model.conditional.neg_log_prob(y[None, :, :], py)
        if model.prior.has_kl:
            KL_q_p = model.prior.step_kl_divergence(q, *prior_params)
        else:
            log_ph = -model.prior.neg_log_prob(h)
            log_qh = -model.posterior.neg_log_prob(h, q[None, :, :])
            KL_q_p = (log_qh - log_ph).mean(axis=0)
        y_energy = -log_py_h.mean(axis=0)

        cost = (y_energy + KL_q_p).mean(axis=0)
        grad = theano.grad(cost, wrt=q, consider_constant=consider_constant)

        cost = y_energy.mean()
        return cost, grad


class MomentumGDIR(GDIR):
    def __init__(self, model, momentum=0.9, name='momentum_GDIR', **kwargs):
        self.momentum = momentum
        super(MomentumGDIR, self).__init__(model, name=name, **kwargs)

    def step_infer(self, epsilon, q, dq_, y, m, *params):
        l = self.inference_rate
        cost, grad = self.e_step(epsilon, q, y, *params)
        dq = (-l * grad + m * dq_).astype(floatX)
        q = (q + dq).astype(floatX)
        return q, dq, cost

    def init_infer(self, q):
        return [T.zeros_like(q)]

    def unpack_infer(self, outs):
        qs, dqs, costs = outs
        return qs, costs

    def params_infer(self):
        return [T.constant(self.momentum).astype(floatX)]


