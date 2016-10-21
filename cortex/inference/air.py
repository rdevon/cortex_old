'''
Adaptive importance sampling inference.
'''

from collections import OrderedDict
import theano
from theano import tensor as T

from .irvi import IRVI, DeepIRVI
from ..utils import concatenate, floatX, scan, slice2
from ..utils.maths import norm_exp, log_mean_exp


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
        q = T.clip(q, 1e-6, 1 - 1e-6)
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
    _required = ['prior', 'conditionals', 'posteriors']
    _components = {
        'prior': {},
        'conditionals': {},
        'posteriors': {}}

    def __init__(self, prior, conditionals, posteriors, name='AIR', **kwargs):
        '''Init function for DeepAIR

        Args:
            name: str
            kwargs: dict, remaining IRVI arguments.
        '''

        models = dict(
            prior=prior,
            conditionals=conditionals,
            posteriors=posteriors)
        
        if len(conditionals) != len(posteriors):
            raise TypeError('Same number of conditionals and posteriors must be'
                            ' provided.')

        super(DeepAIR, self).__init__(name=name, models=models, **kwargs)

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
        qs = []
        rs = []
        hs = []
        start = 0
        for posterior in self.posteriors:
            dim = posterior.dim_out
            q_ = slice2(q, start, start + dim)
            r_ = slice2(r, start, start + dim)
            h = posterior.quantile(r_, q_)
            qs.append(q_)
            rs.append(r_)
            hs.append(h)
            start += dim
        
        weights, cost = self.score(hs, y, qs, *params)

        qs_ = [(weights[:, :, None] * h).sum(axis=0) for h in hs]
        qs  = [inference_rate * qs_[i] + (1 - inference_rate) * qs[i] for i in range(len(qs))]
        qs = [T.clip(q, 1e-6, 1 - 1e-6) for q in qs]
        q = concatenate(qs, axis=1)
        return q, cost

    def score(self, hs, y, qs, *params):
        prior_params = self.select_params('prior', *params)
        log_p = -self.prior.step_neg_log_prob(hs[-1], *prior_params)
        hs = [y[None, :, :]] + hs
        for i in xrange(len(self.conditionals)):
            cond_params = self.select_params(
                'conditionals_{}'.format(i), *params)
            py = self.conditionals[i]._feed(hs[i + 1], *cond_params)['P']
            log_py_h = -self.conditionals[i].neg_log_prob(hs[i], P=py)
            log_qh = -self.posteriors[i].neg_log_prob(
                hs[i + 1], P=qs[i][None, :, :])
            log_p += log_py_h - log_qh
        w_tilde = norm_exp(log_p)
        cost = -log_p.mean()
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

        Es = []
        start = 0
        for posterior in self.posteriors:
            dim = posterior.dim_out
            P_ = slice2(P, start, start + dim)
            E = posterior.generate_random_variables(shape, P=P_)
            Es.append(E)
            start += dim

        return concatenate(Es, axis=len(shape)+1)
    
    def split(self, Q, aslist=False):
        Qs = []
        start = 0
        for i, posterior in enumerate(self.posteriors):
            dim = posterior.dim_out
            Qs.append(slice2(Q, start, start + dim))
            start += dim
        if aslist:
            return Qs
        else:
            return OrderedDict(('Q_{}'.format(i), Q) for i, Q in enumerate(Qs))

    def _sample(self, E, P=None):
        session = self.manager.get_session()
        if P is None:
            if _p(self.name, 'Qk') not in session.tensors.keys():
                raise TypeError('%s.Qk not found in graph nor provided'
                                % self.name)
            P = session.tensors[_p(self.name, 'Qk')]
        Ss = []
        start = 0
        for posterior in self.posteriors:
            dim = posterior.dim_out
            P_ = slice2(P, start, start + dim)
            E_ = slice2(E, start, start + dim)
            Ss.append(posterior._sample(E, P=P))
            
        return concatenate(Ss, axis=E.ndim-1)
    
    def _cost(self, X=None, Qk_samples=None, Q=None, Qk=None):
        params = self.get_params()
        prior_params = self.select_params('prior', *params)
        Qs = self.split(Q, aslist=True)
        Qks = self.split(Qk, aslist=True)
        Qk_samples = self.split(Qk_samples, aslist=True)
        Hs = [X[None, :, :]] + Qk_samples
        
        log_p = -self.prior.step_neg_log_prob(Hs[-1], *prior_params)
        gen_term = log_p.mean()
        cost = log_p.mean()
        kl_term = T.constant(0.).astype(floatX)
        for i in xrange(len(self.conditionals)):
            cond_params = self.select_params(
                'conditionals_{}'.format(i), *params)
            py = self.conditionals[i]._feed(Hs[i + 1], *cond_params)['P']
            log_py_h = -self.conditionals[i].neg_log_prob(Hs[i], P=py)
            log_qh = -self.posteriors[i].neg_log_prob(Hs[i + 1], P=Qs[i][None, :, :])
            log_qhk = -self.posteriors[i].neg_log_prob(Hs[i + 1], P=Qks[i][None, :, :])
            cost += (-log_py_h - log_qh).mean()
            kl_term += (log_qhk - log_qh).mean()
            gen_term += -log_py_h.mean()
            log_p += log_py_h - log_qhk
            
        nll = -log_mean_exp(log_p, axis=0).mean()
        lower_bound = -log_p.mean()
       
        return OrderedDict(cost=cost, kl_term=kl_term, gen_term=gen_term,
                           nll=nll, lower_bound=lower_bound)

_classes = {'AIR': AIR, 'DeepAIR': DeepAIR}