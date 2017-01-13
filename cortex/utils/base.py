'''General tools for cortex.

'''

import numpy as np
import random
import theano
from theano import tensor as T


def resolve_class(cell_type, classes=None):
    from .models import _classes
    if classes is None:
        classes = _classes
    try:
        C = classes[cell_type]
    except KeyError:
        raise KeyError('Unexpected cell subclass `%s`, '
                       'available classes: %s' % (cell_type, classes.keys()))
    return C

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
    if not isinstance(n_steps, int) or n_steps > 1:
        return theano.scan(
            f_scan,
            sequences=seqs,
            outputs_info=outputs_info,
            non_sequences=non_seqs,
            name=name,
            n_steps=n_steps,
            strict=strict
        )
    else:
        outputs_info = [o for o in outputs_info if o is not None]
        if n_steps == 0:
            return outputs_info
        elif n_steps == 1:
            return f_scan(*(seqs + outputs_info + non_seqs))

def tslice(_x, n, dim):
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

def slice2(_x, start, end):
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