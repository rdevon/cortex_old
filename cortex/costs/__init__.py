'''Basic cost functions.

'''

def squared_error(Y_hat, Y):
    if Y_hat.ndim != Y.ndim:
        raise TypeError('Squared error inputs must have the same number of '
                        'dimensions.')
    sq_err = ((Y_hat - Y) ** 2).mean()
    sq_err.name = 'squared_error_loss'
    return sq_err

_costs = {'squared_error': squared_error}