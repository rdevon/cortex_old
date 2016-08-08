'''Basic cost functions.

'''
from ..utils.tools import _p

def squared_error(Y_hat=None, Y=None):
    if Y_hat is None:
        raise TypeError('Y_hat (estimate) must be provided')
    if Y is None:
        raise TypeError('Y (ground truth) must be provided')
    if Y_hat.ndim != Y.ndim:
        raise TypeError('Squared error inputs must have the same number of '
                        'dimensions.')
    sq_err = ((Y_hat - Y) ** 2).sum(-1).mean()
    sq_err.name = 'squared_error_loss'
    return sq_err

def kl_divergence(P=None, Q=None, P_samples=None, cells=None, average=True):
    from .. import _manager as manager

    if P is None:
        raise TypeError('Reference distribution (P) must be provided.')
    if Q is None:
        raise TypeError('Relative distribution (Q) must be provided.')
    if cells is None:
        raise TypeError('In order to properly calculate the KL divergences, '
                        'it is necessary to pass `cells` as a list of the '
                        'cells from which P and Q come from.')
    p_name, q_name = cells

    if isinstance(P, str):
        P_cell = manager.cells[P]
        P = P_cell.get_prob(*P_cell.get_params())
    else:
        P_cell = manager.cells[p_name]

    if isinstance(Q, str):
        Q_cell = manager.cells[Q]
        Q = Q_cell.get_prob(*Q_cell.get_params())
    else:
        Q_cell = manager.cells[q_name]

    if P_cell.has_kl and (
        P_cell.base_distribution == Q_cell.__class__ or
        Q_cell.base_distribution == P_cell.__class__ or
        P_cell_class == Q_cell.__class__):
        kl = P_cell.kl_divergence(
            *(P_cell.split_prob(P) + Q_cell.split_prob(Q)))
        if average:
            return kl.mean()
        else:
            return kl

    else:
        if P_samples is None:
            session = manager.get_session()
            s_name = _p(p_name, 'samples')
            if s_name not in session.tensors.keys():
                raise TypeError('P samples not provided and None found in '
                                'current session.')
            P_samples = session.tensors[s_name]

    neg_term = P_cell.neg_log_prob(P_samples, P=P)
    pos_term = Q_cell.neg_log_prob(P_samples, P=Q)
    kl = pos_term - neg_term

    if average:
        return kl.mean()
    else:
        return kl

_costs = {'squared_error': squared_error, 'kl_divergence': kl_divergence}