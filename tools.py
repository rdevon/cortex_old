"""
Helper module for NMT
"""

from collections import OrderedDict
import numpy as np
import pprint
import random
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import yaml


random_seed = 0xeffe
rng_ = np.random.RandomState(random_seed)

profile = False

f_clip = lambda x, y, z: T.clip(x, y, 1.)

pi = theano.shared(np.pi).astype('float32')

def init_weights(model, weight_noise=False, weight_scale=0.01, dropout=False, **kwargs):
    model.weight_noise = weight_noise
    model.weight_scale = weight_scale
    model.dropout = dropout
    return kwargs

def init_rngs(model, rng=None, trng=None, **kwargs):
    if rng is None:
        rng = rng_
    model.rng = rng
    if trng is None:
        model.trng = RandomStreams(random.randint(0, 10000))
    else:
        model.trng = trng
    return kwargs

def gaussian(x, mu, s):
    return T.exp(-(x - mu)**2 / (2 * s)**2) / (s * T.sqrt(2 * pi)).astype('float32')

def log_gaussian(x, mu, s):
    return -(x - mu)**2 / (2 * s**2) - T.log(s + 1e-7) - T.sqrt(2 * pi).astype('float32')

def load_experiment(experiment_yaml):
    print('Loading experiment from %s' % experiment_yaml)
    exp_dict = yaml.load(open(experiment_yaml))
    print('Experiment hyperparams: %s' % pprint.pformat(exp_dict))
    return exp_dict

def load_model(model_file, f_unpack=None, **extra_args):
    '''
    Loads pretrained model.
    '''

    print 'Loading model from %s' % model_file
    params = np.load(model_file)
    d = dict(params)
    d.update(**extra_args)
    for k, v in d.iteritems():
        try:
            if v == np.array(None, dtype=object):
                d[k] = None
        except ValueError:
            pass

    models, pretrained_kwargs, kwargs = f_unpack(**d)

    print('Pretrained model(s) has the following parameters: \n%s'
          % pprint.pformat(pretrained_kwargs.keys()))

    model_dict = OrderedDict()

    for model in models:
        for k, v in model.params.iteritems():
            try:
                pretrained_v = pretrained_kwargs[
                    '{name}_{key}'.format(name=model.name, key=k)]
                print 'Found %s for %s' % (k, model.name)
                assert model.params[k].shape == pretrained_v.shape, (
                    'Sizes do not match: %s vs %s'
                    % (model.params[k].shape, pretrained_v.shape)
                )
                model.params[k] = pretrained_v
            except KeyError:
                pretrained_v = pretrained_kwargs[
                    '{key}'.format(key=k)]
                print 'Found %s, but name is ambiguous' % (k)
                assert model.params[k].shape == pretrained_v.shape, (
                    'Sizes do not match: %s vs %s'
                    % (model.params[k].shape, pretrained_v.shape)
                )
                model.params[k] = pretrained_v
            except KeyError:
                print '{} not found'.format(k)
        model_dict[model.name] = model

    return model_dict, kwargs

def check_bad_nums(rvals, names):
    for k, out in zip(names, rvals):
        if np.any(np.isnan(out)):
            print k, 'nan'
            return True
        elif np.any(np.isinf(out)):
            print k, 'inf'
            return True
    return False

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
    return W.astype('float32')

def parzen_estimation(samples, tests, h=1.0):
    log_p = 0.
    d = samples.shape[-1]
    z = d * np.log(h * np.sqrt(2 * np.pi))
    for test in tests:
        d_s = (samples - test[None, :]) / h
        e = log_mean_exp((-.5 * d_s ** 2).sum(axis=1), as_numpy=True, axis=0)
        log_p += e - z
    return log_p / float(tests.shape[0])

def tanh(x):
    return T.tanh(x)

def linear(x):
    return x

def log_mean_exp(x, axis=None, as_numpy=False):
    if as_numpy:
        Te = np
    else:
        Te = T
    x_max = Te.max(x, axis=axis, keepdims=True)
    return Te.log(Te.mean(Te.exp(x - x_max), axis=axis, keepdims=True)) + x_max

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