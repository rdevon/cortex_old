'''
Gradient-Descent Iterative Refinement
'''

from collections import OrderedDict
import theano
from theano import tensor as T

from .irvi import IRVI, DeepIRVI
from ..utils import concatenate, floatX, scan, slice2
from ..utils.maths import norm_exp, log_mean_exp, log_sum_exp
from ..utils.tools import update_dict_of_lists


class GDIR(IRVI):
    _required = ['prior', 'conditional', 'posterior']
    _components = {
        'prior': {},
        'conditional': {},
        'posterior': {}}
    
    def __init__(self, prior, conditional, posterior, pass_gradients=False,
                 name='GDIR', **kwargs):
        
        models = dict(
            prior=prior,
            conditional=conditional,
            posterior=posterior)

        super(GDIR, self).__init__(models=models, name=name,
                                   pass_gradients=pass_gradients, **kwargs)

    def e_step(self, r, q, y, inference_rate, *params):
        prior_params = self.select_params('prior', *params)
        cond_params = self.select_params('conditional', *params)

        h = self.posterior.quantile(r, q)
        py = self.conditional._feed(h, *cond_params)['P']
        log_py_h = -self.conditional.neg_log_prob(y[None, :, :], P=py)

        consider_constant = [y] + list(params)
        
        if self.prior.has_kl:
            q_params = self.posterior.distribution.split_prob(q)
            KL_q_p = self.posterior.kl_divergence(
                *(q_params + prior_params))[None, :]
        else:
            log_ph   = -self.prior.step_neg_log_prob(h, *prior_params)
            log_qh   = -self.posterior.neg_log_prob(h, P=q[None, :, :])
            KL_q_p = log_qh - log_ph
            
        log_p = log_py_h - KL_q_p

        cost = -log_p.mean()
        grad = theano.grad(cost, wrt=q, consider_constant=consider_constant)
        
        return log_p.mean(), grad
    
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
    
    def _cost(self, X=None, Qk_samples=None, Q=None, Qk=None,
              vae=False, return_stats=False):
        params = self.get_params()
        return self.get_stats(X, Qk_samples, Q, Qk, vae, return_stats, *params)
    
    def get_stats(self, X, Qk_samples, Q, Qk, vae, return_stats, *params):
        prior_params = self.select_params('prior', *params)
        cond_params = self.select_params('conditional', *params)
        
        py = self.conditional._feed(Qk_samples, *cond_params)['P']
        gen_term = self.conditional.neg_log_prob(X[None, :, :], P=py)
        gen_term += self.prior.neg_log_prob(Qk_samples)
        infer_term = self.posterior.neg_log_prob(Qk_samples, P=Q[None, :, :])
        infer_termk = self.posterior.neg_log_prob(Qk_samples, P=Qk[None, :, :])
        
        kl_term = (infer_term - infer_termk).mean()
            
        log_p = -gen_term + infer_termk
        log_p0 = -gen_term + infer_term
        
        w_tilde = norm_exp(log_p)
        ess = (1. / (w_tilde ** 2).sum(0)).mean()
        log_ess = (-T.log((w_tilde ** 2).sum(0))).mean()
        
        if vae:
            cost = -log_p.mean()
        else:
            cost = (gen_term + infer_term).mean()
            
        nll = -log_mean_exp(log_p, axis=0).mean()
        lower_bound = -log_p.mean()

        if return_stats:        
            return OrderedDict(kl_term=kl_term, gen_term=gen_term.mean(),
                               nll=nll, lower_bound=lower_bound,
                               infer_term=infer_term.mean(),
                               infer_termk=infer_termk.mean(),
                               ess=ess, log_ess=log_ess, Q_mean=Q.mean(),
                               Q_sam_mean=Qk_samples.mean())
        else:
            return OrderedDict(cost=cost, constants=[w_tilde])

class MomentumGDIR(GDIR):
    def __init__(self, prior, conditional, posterior, momentum=0.9,
                 name='momentum_GDIR', **kwargs):
        
        self.momentum = momentum
        super(MomentumGDIR, self).__init__(prior, conditional, posterior,
                                           name=name, **kwargs)

    def step_infer(self, r, q, dq_, y, momentum, inference_rate, *params):
        cost, grad = self.e_step(r, q, y, inference_rate, *params)
        dq = (-inference_rate * grad + momentum * dq_).astype(floatX)
        q = q + dq
        extra = grad
        return q, dq, cost, extra

    def init_infer(self, q):
        return [T.zeros_like(q)]

    def unpack_infer(self, outs):
        qs, dqs, costs, extra = outs
        return qs, costs, extra

    def params_infer(self, inference_rate):
        return [self.momentum, inference_rate]

_classes = {'MomentumGDIR': MomentumGDIR}
