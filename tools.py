"""
Helper module for NMT
"""

from collections import OrderedDict
import numpy as np
import theano
from theano import tensor as T


random_seed = 0xeffe
rng_ = np.random.RandomState(random_seed)

profile = False

f_clip = lambda x, y, z: T.clip(x, y, 1.)

pi = theano.shared(np.pi).astype('float32')

def gaussian(x, mu, s):
    return T.exp(-(x - mu)**2 / (2 * s)**2) / (s * T.sqrt(2 * pi)).astype('float32')

def log_gaussian(x, mu, s):
    return -(x - mu)**2 / (2 * s**2) - T.log(s + 1e-7) - T.sqrt(2 * pi).astype('float32')

def check_bad_nums(rval_dict, i):
    bad_nums = OrderedDict()
    for k, rval in rval_dict.iteritems():
        if np.any(np.isnan(rval)):
            bad_nums[k] = 'Nan'
        if np.any(np.isinf(rval)):
            bad_nums[k] = 'Inf'
    if len(bad_nums.keys()) > 0:
        raise ValueError('Inf or nans detected at %d: %s'
                         % (i, bad_nums))

def flatten_dict(d):
    rval = OrderedDict()
    for k, v in d.iteritems():
        if isinstance(v, OrderedDict):
            new_d = flatten_dict(v)
            for k_, v_ in new_d.iteritems():
                rval[k + '_' + k_] = v_
        else:
            rval[k] = v
    return rval

# push parameters to Theano shared variables
def zipp(params, tparams):
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)

# pull parameters from Theano shared variables
def unzip(zipped):
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params

# get the list of parameters: Note that tparams must be OrderedDict
def itemlist(tparams):
    return tparams.values()

# make prefix-appended name
def _p(pp, name):
    return '%s_%s'%(pp, name)

def ortho_weight(ndim, rng=None):
    if not rng:
        rng = rng_
    W = rng.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype('float32')

def norm_weight(nin, nout=None, scale=0.01, ortho=True, rng=None):
    if not rng:
        rng = rng_
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin, rng=rng)
    else:
        W = scale * rng.randn(nin, nout)
    return W.astype('float32')# some utilities

def tanh(x):
    return T.tanh(x)

def linear(x):
    return x

def concatenate(tensor_list, axis=0):
    """
    Alternative implementation of `theano.T.concatenate`.
    This function does exactly the same thing, but contrary to Theano's own
    implementation, the gradient is implemented on the GPU.
    Backpropagating through `theano.tensor.concatenate` yields slowdowns
    because the inverse operation (splitting) needs to be done on the CPU.
    This implementation does not have that problem.
    :usage:
        >>> x, y = theano.tensor.matrices('x', 'y')
        >>> c = concatenate([x, y], axis=1)
    :parameters:
        - tensor_list : list
            list of Theano tensor expressions that should be concatenated.
        - axis : int
            the tensors will be joined along this axis.
    :returns:
        - out : tensor
            the concatenated tensor expression.
    """
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