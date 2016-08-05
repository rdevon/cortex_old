'''Basic visualization ops.

'''

from ..utils.tools import get_srng


def random_set(n_samples=None, **kwargs):
    X = kwargs.values()[0]
    srng = get_srng()
    idx = srng.permutation(n=X.shape[0], size=(1,))[0][:n_samples]
    outs = dict(idx=idx)
    for k, v in kwargs.iteritems():
        outs[k] = v[idx]
    return outs

_ops = {'random_set': random_set}