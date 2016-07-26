'''Init for utils.

'''

import numpy as np
import random
import theano
from theano import tensor as T
import warnings


warnings.filterwarnings('ignore', module='nipy')
warnings.filterwarnings('ignore',
                        'This call to matplotlib.use() has no effect.*',
                        UserWarning)
intX = 'int64'
floatX = theano.config.floatX
pi = theano.shared(np.pi).astype(floatX)
e = theano.shared(np.e).astype(floatX)
_random_seed = random.randint(1, 10000)
_rng = np.random.RandomState(_random_seed)

def concatenate(tensor_list, axis=0):
    '''Alternative implementation of `theano.T.concatenate`.

    This function does exactly the same thing, but contrary to Theano's own
    implementation, the gradient is implemented on the GPU.
    Backpropagating through `theano.tensor.concatenate` yields slowdowns
    because the inverse operation (splitting) needs to be done on the CPU.
    This implementation does not have that problem.

    Examples:
        >>> x, y = theano.tensor.matrices('x', 'y')
        >>> c = concatenate([x, y], axis=1)

    Args:
        tensor_list (list): list of Theano tensor expressions that should be concatenated.
        axis (int): the tensors will be joined along this axis.

    Returns:
        T.tensor: the concatenated tensor expression.

    From Cho's arctic repo.

    '''
    concat_size = sum(tt.shape[axis] for tt in tensor_list)

    output_shape = ()
    for k in range(axis):
        output_shape += (tensor_list[0].shape[k],)
    output_shape += (concat_size,)
    for k in range(axis + 1, tensor_list[0].ndim):
        output_shape += (tensor_list[0].shape[k],)

    out = T.zeros(output_shape)
    offset = 0
    for tt in tensor_list:
        indices = ()
        for k in range(axis):
            indices += (slice(None),)
        indices += (slice(offset, offset + tt.shape[axis]),)
        for k in range(axis + 1, tensor_list[0].ndim):
            indices += (slice(None),)

        out = T.set_subtensor(out[indices], tt)
        offset += tt.shape[axis]

    return out

def scan(f_scan, seqs, outputs_info, non_seqs, n_steps, name='scan',
         strict=False):
    '''Convenience function for scan.

    Args:
        f_scan (function): scanning function.
        seqs (list or tuple): list of sequence tensors.
        outputs_info (list or tuple): list of scan outputs.
        non_seqs (list or tuple): list of non-sequences.
        n_steps (int): number of steps.
        name (str): name of scanning procedure.
        strict (bool).

    Returns:
        tuple: scan outputs.
        theano.OrderedUpdates: updates.

    '''
    return theano.scan(
        f_scan,
        sequences=seqs,
        outputs_info=outputs_info,
        non_sequences=non_seqs,
        name=name,
        n_steps=n_steps,
        strict=strict
    )

def _slice(_x, n, dim):
    '''Slice a tensor into 2 along last axis.

    Extended from Cho's arctic repo.

    Args:
        _x (T.tensor).
        n (int).
        dim (int).

    Returns:
        T.tensor.

    '''
    if _x.ndim == 1:
        return _x[n*dim:(n+1)*dim]
    elif _x.ndim == 2:
        return _x[:, n*dim:(n+1)*dim]
    elif _x.ndim == 3:
        return _x[:, :, n*dim:(n+1)*dim]
    elif _x.ndim == 4:
        return _x[:, :, :, n*dim:(n+1)*dim]
    else:
        raise ValueError('Number of dims (%d) not supported'
                         ' (but can add easily here)' % _x.ndim)

def _slice2(_x, start, end):
    '''Slightly different slice function than above.

    Args:
        _x (T.tensor).
        start (int).
        end (int).

    Returns:
        T.tensor.

    '''
    if _x.ndim == 1:
        return _x[start:end]
    elif _x.ndim == 2:
        return _x[:, start:end]
    elif _x.ndim == 3:
        return _x[:, :, start:end]
    elif _x.ndim == 4:
        return _x[:, :, :, start:end]
    else:
        raise ValueError('Number of dims (%d) not supported'
                         ' (but can add easily here)' % _x.ndim)