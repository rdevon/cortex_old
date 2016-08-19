'''Math functions.

'''

from math import sqrt, floor
from theano import tensor as T

from . import floatX


def parzen_estimation(samples, tests, h=1.0):
    '''Estimate parzen window.

    '''
    log_p = 0.
    d = samples.shape[-1]
    z = d * np.log(h * np.sqrt(2 * np.pi))
    for test in tests:
        d_s = (samples - test[None, :]) / h
        e = log_mean_exp((-.5 * d_s ** 2).sum(axis=1), as_numpy=True, axis=0)
        log_p += e - z
    return log_p / float(tests.shape[0])

def norm_exp(log_factor):
    '''Gets normalized weights.

    '''
    log_factor = log_factor - T.log(log_factor.shape[0]).astype(floatX)
    w_norm   = log_sum_exp(log_factor, axis=0)
    log_w    = log_factor - T.shape_padleft(w_norm)
    w_tilde  = T.exp(log_w)
    return w_tilde

def log_mean_exp(x, axis=None, as_numpy=False):
    '''Numerically stable log(exp(x).mean()).

    '''
    Te = np if as_numpy else T
    x_max = Te.max(x, axis=axis, keepdims=True)
    return Te.log(Te.mean(Te.exp(x - x_max), axis=axis, keepdims=True)) + x_max

def log_sum_exp(x, axis=None):
    '''Numerically stable log( sum( exp(A) ) ).

    '''
    x_max = T.max(x, axis=axis, keepdims=True)
    y = T.log(T.sum(T.exp(x - x_max), axis=axis, keepdims=True)) + x_max
    y = T.sum(y, axis=axis)
    return y

def logit(z):
    '''Logit function.

    :math:`\log \\frac{z}{1 - z}`

    Args:
        z (T.tensor).

    Returns:
        T.tensor.

    '''
    z = T.clip(z, 1e-7, 1.0 - 1e-7)
    return T.log(z) - T.log(1 - z)

def shuffle_columns(x, srng):
    '''Shuffles a tensor along the second index.

    Args:
        x (T.tensor).
        srng (sharedRandomstream).

    '''
    def step_shuffle(m, perm):
        return m[perm]

    perm_mat = srng.permutation(n=x.shape[0], size=(x.shape[1],))
    y, _ = scan(
        step_shuffle, [x.transpose(1, 0, 2), perm_mat], [None], [], x.shape[1],
        name='shuffle', strict=False)
    return y.transpose(1, 0, 2)

def split_int_into_closest_two(x):
    '''Splits an integer into the closest 2 integers.

    Args:
        x (int).

    Returns:
        int.

    Raises:
        ValueError: if input is not an integer.

    '''

    if not isinstance(x, (int, long)):
        raise ValueError('Input is not an integer.')

    n = floor(sqrt(x))
    while True:
        if n < 1:
            raise ValueError
        rem = x % n
        if rem == 0:
            return int(n), int(x / n)
        n -= 1