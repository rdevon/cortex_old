'''Module for noise operations.

'''

from .. import utils


def binary(X=None, noise=0.5):
    trng = utils.tools.get_trng()
    noise = trng.binomial(p=(1 - noise), size=X.shape, n=1, dtype=X.dtype)
    output = noise * X
    return output

_ops = {'binary': binary}