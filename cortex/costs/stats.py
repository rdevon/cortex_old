'''Basic stats

'''

from collections import OrderedDict
from theano import tensor as T

from .basic import kl_divergence


def logistic_regression_stats(P=None, Y=None):
    stats = OrderedDict()
    Y_pred = T.argmax(P, axis=-1)
    Y_ = T.argmax(Y, axis=-1)
    stats['error'] = T.neq(Y_pred, Y_).mean()
    P = T.clip(P, 1e-7, 1 - 1e-7)
    stats['nll'] = (
        -Y * T.log(P) - (1 - Y) * T.log(1 - P)).sum(axis=-1).mean()
    return stats

def variational_inference(X=None, conditional=None, posterior=None, prior=None,
                          posterior_samples=None, cells=None):
    from .. import _manager as manager

    kl_term = kl_divergence(P=posterior, Q=prior, P_samples=posterior_samples,
                            cells=cells[1:], average=False)

    def get_cell(P, name):
        if isinstance(P, str):
            cell = manager.cells[P]
            P = cell.get_prob(*cell.get_params())
        else:
            cell = manager.cells[name]

        return P, cell

    conditional, cond_cell = get_cell(conditional, cells[0])

    nll_term = cond_cell.neg_log_prob(X, P=conditional)
    log_p = nll_term[None, :, :] + kl_term
    lower_bound = -log_p.mean()
    log_p_max = T.max(log_p, axis=0, keepdims=True)
    w         = T.exp(log_p - log_p_max)
    nll = (T.log(w.mean(axis=0, keepdims=True)) + log_p_max).mean()
    return dict(lower_bound=lower_bound, negative_log_likelihood=nll)

_stats = {'logistic_regression': logistic_regression_stats,
          'variational_inference': variational_inference}
