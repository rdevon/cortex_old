'''
Adaptive importance sampling inference.
'''

from collections import OrderedDict
import theano
from theano import tensor as T

from .irvi import IRVI, DeepIRVI
from ..utils import concatenate, floatX, scan, slice2
from ..utils.maths import norm_exp, log_mean_exp, log_sum_exp
from ..utils.tools import update_dict_of_lists

class AIR(IRVI):
    '''
    Adaptive importance refinement (AIR).

    Inference procedure to refine the posterior using adaptive importance
    sampling (AIS)
    '''
    _required = ['prior', 'conditional', 'posterior']
    _args = ['bidirectional']
    _components = {
        'prior': {},
        'conditional': {},
        'posterior': {}}

    def __init__(self, prior, conditional, posterior, bidirectional=False,
                 name='AIR', **kwargs):
        '''Init function for AIR

        Args:
            name: str
            kwargs: dict, remaining IRVI arguments.
        '''

        models = dict(
            prior=prior,
            conditional=conditional,
            posterior=posterior)
        self.bidirectional = bidirectional

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
        weights, cost, extra = self.score(h, y, q, *params)

        q_ = (weights[:, :, None] * h).sum(axis=0)
        q  = inference_rate * q_ + (1 - inference_rate) * q
        q = T.clip(q, 1e-7, 1 - 1e-7)
        return q, cost, extra

    def score(self, h, y, q, *params):
        prior_params = self.select_params('prior', *params)
        post_params = self.select_params('posterior', *params)
        cond_params = self.select_params('conditional', *params)
        py       = self.conditional._feed(h, *cond_params)['P']
        log_py_h = -self.conditional.neg_log_prob(y[None, :, :], P=py)
        log_ph   = -self.prior.step_neg_log_prob(h, *prior_params)
        log_qh   = -self.posterior.neg_log_prob(h, P=q[None, :, :])
        log_p     = log_py_h + log_ph - log_qh
        if self.bidirectional:
            w_tilde = norm_exp(0.5 * log_p)
        else:
            w_tilde = norm_exp(log_p)
        cost    = -log_p.mean()
        return w_tilde, -log_p.mean(), py

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
    _args = ['bidirectional']

    def __init__(self, prior, conditionals, posteriors, bidirectional=False,
                 name='AIR', **kwargs):
        '''Init function for DeepAIR

        Args:
            name: str
            kwargs: dict, remaining IRVI arguments.
        '''

        models = dict(
            prior=prior,
            conditionals=conditionals,
            posteriors=posteriors)
        self.bidirectional = bidirectional
        
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
        hs = []
        start = 0
        qs = self.split(q, aslist=True)
        rs = self.split(r, aslist=True)
        hs = [self.posteriors[i].quantile(rs[i], qs[i])
              for i in range(len(self.posteriors))]
                
        weights, cost, extra = self.score(hs, y, qs, *params)

        qs_ = [(weights[:, :, None] * h).sum(axis=0) for h in hs]
        qs  = [inference_rate * qs_[i] + (1 - inference_rate) * qs[i]
               for i in range(len(qs))]
        qs = [T.clip(q, 1e-6, 1 - 1e-6) for q in qs]
        q = concatenate(qs, axis=1)
        return q, cost, hs[-1]

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

        if self.bidirectional:
            log_p *= 0.5
        
        w_tilde = norm_exp(log_p)
        cost = -log_p.mean()
        
        return w_tilde, cost, hs[-1]

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

        Ps = self.split(P, aslist=True)
        Es = [self.posteriors[i].generate_random_variables(shape, P=Ps[i])
              for i in range(len(self.posteriors))]

        return concatenate(Es, axis=P.ndim-1+len(shape))
    
    def split(self, Q, aslist=False):
        Qs = []

        start = 0
        for posterior in self.posteriors:
            dim = posterior.mlp.dim_out
            Qs.append(slice2(Q, start, start + dim))
            start += dim

        if aslist:
            return Qs
        else:
            return OrderedDict(('Q_{}'.format(i), Q) for i, Q in enumerate(Qs))

    def _sample(self, E, P=None):
        '''
        session = self.manager.get_session()
        if P is None:
            if _p(self.name, 'Qk') not in session.tensors.keys():
                raise TypeError('%s.Qk not found in graph nor provided'
                                % self.name)
            P = session.tensors[_p(self.name, 'Qk')]
        '''
        start = 0
        Ps = self.split(P, aslist=True)
        Es = self.split(E, aslist=True)
        Ss = [self.posteriors[i].quantile(Es[i], Ps[i])
              for i in range(len(self.posteriors))]
            
        return concatenate(Ss, axis=E.ndim-1)

    def _cost(self, X=None, Qk_samples=None, Q=None, Qk=None,
              reweight_posterior=False, reweight_conditional=False,
              return_stats=False):
        params = self.get_params()
        return self.get_stats(X, Qk_samples, Q, Qk,
                              reweight_posterior, reweight_conditional,
                              return_stats, *params)
        
    def get_stats(self, X, Qk_samples, Q, Qk,
                  reweight_posterior, reweight_conditional,
                  return_stats, *params):
        prior_params = self.select_params('prior', *params)

        Qs = self.split(Q, aslist=True)
        Qks = self.split(Qk, aslist=True)
        Qk_samples = self.split(Qk_samples, aslist=True)
        Hs = [X[None, :, :]] + Qk_samples
        
        gen_term = self.prior.step_neg_log_prob(Hs[-1], *prior_params)
        infer_term = T.zeros_like(gen_term)
        infer_termk = T.zeros_like(gen_term)
        kl_term = T.constant(0.).astype(floatX)
        for i in xrange(len(self.conditionals)):
            cond_params = self.select_params(
                'conditionals_{}'.format(i), *params)
            py = self.conditionals[i]._feed(Hs[i + 1], *cond_params)['P']
            nl_py_h = self.conditionals[i].neg_log_prob(Hs[i], P=py)

            nl_qh = self.posteriors[i].neg_log_prob(Hs[i + 1], P=Qs[i][None, :, :])
            nl_qhk = self.posteriors[i].neg_log_prob(Hs[i + 1], P=Qks[i][None, :, :])
            
            kl_term += (nl_qh - nl_qhk).mean()
            gen_term += nl_py_h
            infer_term += nl_qh
            infer_termk += nl_qhk
            
        log_p = -gen_term + infer_termk
        w_tilde = norm_exp(log_p)
        ess = (1. / (w_tilde ** 2).sum(0)).mean()
        log_ess = (-T.log((w_tilde ** 2).sum(0))).mean()
        
        cost = T.constant(0.).astype(floatX)
        if reweight_conditional:
            cost += (w_tilde * gen_term).sum(0).sum(0)
        else:
            cost += gen_term.mean(0).sum(0)
        if reweight_posterior:
            cost += (w_tilde * infer_term).sum(0).sum(0)
        else:
            cost += infer_term.mean(0).sum(0)
            
        nll = -log_mean_exp(log_p, axis=0).mean()
        lower_bound = -log_p.mean()

        if return_stats:        
            return OrderedDict(kl_term=kl_term, gen_term=gen_term.mean(),
                               nll=nll, lower_bound=lower_bound,
                               infer_term=infer_termk.mean(),
                               ess=ess, log_ess=log_ess)
        else:
            return OrderedDict(cost=cost, constants=[w_tilde])

    def test(self, X=None, Q=None, Qs=None, n_steps=None, n_samples=None):
        rvals = OrderedDict()
        params = self.get_params()
        Es = self.generate_random_variables((n_samples, n_steps + 1), P=Q)
        Qk_samples = self._sample(Es, P=Qs).transpose(1, 0, 2, 3)
        stat_keys = ['kl_term', 'gen_term', 'nll', 'lower_bound', 'infer_term',
                     'ess', 'log_ess']
        
        seqs = [Qs, Qk_samples]
        outputs_info = [None] * len(stat_keys)
        non_seqs = [Q, X] + list(params) 
        def step(Qk, samples, Q, X, *params):
            outs = self.get_stats(X, samples, Q, Qk, False, False, True, *params)
            return [outs[k] for k in stat_keys]

        rvals, updates = scan(step, seqs, outputs_info, non_seqs,
                              n_steps+1, self.name + '_test')
            
        return OrderedDict((k, v) for k, v in zip(stat_keys, rvals))

_classes = {'AIR': AIR, 'DeepAIR': DeepAIR}
