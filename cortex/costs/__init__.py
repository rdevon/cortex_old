'''Basic cost functions.

'''

def squared_error(Y_hat=None, Y=None):
    if Y_hat is None:
        raise TypeError('Y_hat (estimate) must be provided')
    if Y is None:
        raise TypeError('Y (ground truth) must be provided')
    if Y_hat.ndim != Y.ndim:
        raise TypeError('Squared error inputs must have the same number of '
                        'dimensions.')
    sq_err = ((Y_hat - Y) ** 2).mean()
    sq_err.name = 'squared_error_loss'
    return sq_err

_costs = {'squared_error': squared_error}