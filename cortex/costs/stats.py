'''Basic stats

'''

from collections import OrderedDict
from theano import tensor as T

from .basic import kl_divergence
from ..utils.maths import log_mean_exp


def basic_stats(X=None):
    mean = X.mean()
    std = X.std()
    mi = X.min()
    ma = X.max()
    stats = OrderedDict(mean=mean, std=std, min=mi, max=ma)
    return stats

def cell_stats(cell):
    from .. import _manager as manager
    stats = OrderedDict()
    for k, v in manager.tparams.iteritems():
        split = k.split('.')
        cell_name = '.'.join(split[:-1])
        t_name = split[-1]
        if cell_name == cell:
            t_stats = basic_stats(X=v)
            stats.update(**dict(('.'.join([cell_name, t_name, k]), v) for k, v in t_stats.iteritems()))

    return stats

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
    posterior, post_cell = get_cell(posterior, cells[1])
    prior, prior_cell = get_cell(prior, cells[2])

    prior_entropy = prior_cell.entropy()
    posterior_entropy = post_cell.entropy(P=posterior)

    nll_term = cond_cell.neg_log_prob(X, P=conditional)
    log_p = nll_term[None, :, :] + kl_term
    lower_bound = -log_p.mean()
    nll = -log_mean_exp(log_p, axis=0).mean()

    rvals = OrderedDict()
    rvals['prior_entropy'] = prior_entropy
    rvals['posterior_entropy'] = posterior_entropy
    rvals['kl_term'] = kl_term.mean()
    rvals['lower_bound'] = lower_bound
    rvals['negative_log_likelihood'] = nll
    return rvals

_stats = {'logistic_regression': logistic_regression_stats,
          'variational_inference': variational_inference,
          'basic_stats': basic_stats,
          'cell_stats': cell_stats}
