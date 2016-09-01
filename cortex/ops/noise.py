'''Module for noise operations.

'''

from .. import utils


def binary(X=None, noise=0.5):
    from ..manager import get_manager
    manager = get_manager()

    if manager.get_session().noise:
        trng = utils.tools.get_trng()
        noise = trng.binomial(p=(1 - noise), size=X.shape, n=1, dtype=X.dtype)
        output = noise * X
    else:
        output = X
    return output

def gaussian(X=None, noise=0.01):
    from ..manager import get_manager
    manager = get_manager()

    if manager.get_session().noise:
        trng = utils.tools.get_trng()
        noise = trng.normal(avg=0, std=noise, size=X.shape, dtype=X.dtype)
        output = noise + X
    else:
        output = X
    return output

_ops = {'binary': binary, 'gaussian': gaussian}