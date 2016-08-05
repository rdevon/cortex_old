'''General visualization for classifiers.

'''

from theano import tensor as T

from ..utils.tools import get_srng


def top_errors(P=None, Y=None, n_samples=None):
    Y_pred = T.argmax(P, axis=-1)
    Y_ = T.argmax(Y, axis=-1)
    errors = T.neq(Y_pred, Y_)

    n_ = errors.sum()
    if n_samples is None:
        n_samples = n_
    n_samples = T.switch(T.gt(n_samples, n_), n_samples, n_)

    P_max = T.max(P, axis=-1)
    Z = P_max * errors
    idx = T.argsort(Z)[::-1][:n_errors]
    return dict(idx=idx, Y_pred=Y_pred[idx], P=P_max[idx])

def random_set(P=None, Y=None, X=None, n_samples=None):
    Y_pred = T.argmax(P, axis=-1)
    Y_ = T.argmax(Y, axis=-1)
    P_max = T.max(P, axis=-1)
    srng = get_srng()
    idx = srng.permutation(n=P.shape[0], size=(1,))[0][:n_samples]
    return dict(idx=idx, Y_pred=Y_pred[idx], P=P_max[idx], Y=Y_[idx], X=X[idx])

_ops = {'top_errors': top_errors, 'random_set': random_set}