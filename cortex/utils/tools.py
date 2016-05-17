"""
Helper module for learning.
"""

from collections import OrderedDict
from ConfigParser import ConfigParser
import numpy as np
import os
import pprint
from progressbar import (
    Bar,
    Percentage,
    ProgressBar,
    RotatingMarker,
    SimpleProgress,
    Timer
)
import random
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.shared_randomstreams import RandomStreams as SRandomStreams
import warnings
import yaml

from cortex.utils import floatX


random_seed = random.randint(0, 10000)
rng_ = np.random.RandomState(random_seed)
profile = False

# For getting terminal column width
_, _columns = os.popen('stty size', 'r').read().split()
_columns = int(_columns)

def print_section(s):
    '''For printing sections to scripts nicely.'''
    print ('-' * 3) + s + ('-' * (_columns - 3 - len(s)))

def get_paths():
    '''Pulls all paths from `paths.conf` file.'''
    d = os.path.expanduser('~')
    config_file = os.path.join(d, '.cortexrc')
    config = ConfigParser()
    config.read(config_file)
    path_dict = config._sections['PATHS']
    path_dict.pop('__name__')
    return path_dict

def resolve_path(p):
    '''Resolves a path using the `paths.conf` file.'''
    path_dict = get_paths()
    for k, v in path_dict.iteritems():
        p = p.replace(k, v)
    return p

def get_srng():
    '''Shared Randomstream'''
    srng = SRandomStreams(random.randint(0, 1000000))
    return srng

def get_trng():
    '''Normal Randomstream'''
    trng = RandomStreams(random.randint(0, 1000000))
    return trng

def warn_kwargs(c, **kwargs):
    if len(kwargs) > 0:
        warnings.warn('Class instance %s has leftover kwargs %s'
                       % (type(c), kwargs), RuntimeWarning)

def update_dict_of_lists(d_to_update, **d):
    '''Updates a dict of list with kwargs.'''
    for k, v in d.iteritems():
        if k in d_to_update.keys():
            d_to_update[k].append(v)
        else:
            d_to_update[k] = [v]

def debug_shape(X, x, t_out, updates=None):
    '''Debug function that returns the shape then raises assert error.'''
    f = theano.function([X], t_out, updates=updates)
    out = f(x)
    print out.shape
    assert False

def print_profile(tparams):
    '''Prints shapes of the shared variables.'''
    print 'Print profile for tparams (name, shape)'
    for (k, v) in tparams.iteritems():
        print '\t', k, v.get_value().shape

def shuffle_columns(x, srng):
    '''Shuffles a tensor along the second index.'''
    def step_shuffle(m, perm):
        return m[perm]

    perm_mat = srng.permutation(n=x.shape[0], size=(x.shape[1],))
    y, _ = scan(
        step_shuffle, [x.transpose(1, 0, 2), perm_mat], [None], [], x.shape[1],
        name='shuffle', strict=False)
    return y.transpose(1, 0, 2)

def scan(f_scan, seqs, outputs_info, non_seqs, n_steps, name='scan',
         strict=False):
    '''Convenience function for scan.'''
    return theano.scan(
        f_scan,
        sequences=seqs,
        outputs_info=outputs_info,
        non_sequences=non_seqs,
        name=name,
        n_steps=n_steps,
        profile=profile,
        strict=strict
    )

def init_weights(model, weight_noise=False, weight_scale=0.001, dropout=False,
                 **kwargs):
    '''Inialization function for weights.'''
    model.weight_noise = weight_noise
    model.weight_scale = weight_scale
    model.dropout = dropout
    return kwargs

def init_rngs(model, rng=None, trng=None, **kwargs):
    '''Initialization function for RNGs.'''
    if rng is None:
        rng = rng_
    model.rng = rng
    if trng is None:
        model.trng = RandomStreams(random.randint(0, 10000))
    else:
        model.trng = trng
    return kwargs

def logit(z):
    '''Logit function.'''
    z = T.clip(z, 1e-7, 1.0 - 1e-7)
    return T.log(z) - T.log(1 - z)

def _slice(_x, n, dim):
    '''Slice a tensor into 2 along last axis.

    Extended from Cho's arctic repo.
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
    '''Slightly different slice function than above.'''
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

def load_experiment(experiment_yaml):
    '''Load an experiment from a yaml.'''
    print('Loading experiment from %s' % experiment_yaml)
    exp_dict = yaml.load(open(experiment_yaml))
    print('Experiment hyperparams: %s' % pprint.pformat(exp_dict))
    return exp_dict

def load_model(model_file, f_unpack=None, strict=True, **extra_args):
    '''Loads pretrained model.'''

    print 'Loading model from %s' % model_file
    params = np.load(model_file)
    d = dict()
    for k in params.keys():
        try:
            d[k] = params[k].item()
        except ValueError:
            d[k] = params[k]

    d.update(**extra_args)
    models, pretrained_kwargs, kwargs = f_unpack(**d)

    print('Pretrained model(s) has the following parameters: \n%s'
          % pprint.pformat(pretrained_kwargs.keys()))

    model_dict = OrderedDict()

    for model in models:
        print '--- Loading params for %s' % model.name
        for k, v in model.params.iteritems():
            try:
                param_key = '{name}_{key}'.format(name=model.name, key=k)
                pretrained_v = pretrained_kwargs.pop(param_key)
                print 'Found %s for %s %s' % (k, model.name, pretrained_v.shape)
                assert model.params[k].shape == pretrained_v.shape, (
                    'Sizes do not match: %s vs %s'
                    % (model.params[k].shape, pretrained_v.shape)
                )
                model.params[k] = pretrained_v
            except KeyError:
                try:
                    param_key = '{key}'.format(key=k)
                    pretrained_v = pretrained_kwargs[param_key]
                    print 'Found %s, but name is ambiguous' % (k)
                    assert model.params[k].shape == pretrained_v.shape, (
                        'Sizes do not match: %s vs %s'
                        % (model.params[k].shape, pretrained_v.shape)
                    )
                    model.params[k] = pretrained_v
                except KeyError:
                    print '{} not found'.format(k)
        model_dict[model.name] = model

    if len(pretrained_kwargs) > 0 and strict:
        raise ValueError('ERROR: Leftover params: %s' %
                         pprint.pformat(pretrained_kwargs.keys()))
    elif len(pretrained_kwargs) > 0:
        warnings.warn('Leftover params: %s' %
                      pprint.pformat(pretrained_kwargs.keys()))

    return model_dict, kwargs

def check_bad_nums(rvals, names):
    '''Checks for nans and infs.'''
    found = False
    for k, out in zip(names, rvals):
        if np.any(np.isnan(out)):
            print 'Found nan num ', k, '(nan)'
            found = True
        elif np.any(np.isinf(out)):
            print 'Found inf ', k, '(inf)'
            found = True
    return found

def flatten_dict(d):
    '''Flattens a dictionary of dictionaries.'''
    rval = OrderedDict()
    for k, v in d.iteritems():
        if isinstance(v, OrderedDict):
            new_d = flatten_dict(v)
            for k_, v_ in new_d.iteritems():
                rval[k + '_' + k_] = v_
        else:
            rval[k] = v
    return rval

def zipp(params, tparams):
    '''Push parameters to Theano shared variables.

    From Cho's arctic repo.
    '''
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)

def unzip(zipped):
    '''Pull parameters from Theano shared variables.

    From Cho's arctic repo.
    '''
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params

def itemlist(tparams):
    '''Get the list of parameters: Note that tparams must be OrderedDict.

    From Cho's arctic repo.
    '''
    return tparams.values()

def _p(pp, name):
    '''Make prefix-appended name

    From Cho's arctic repo.
    '''
    return '%s_%s'%(pp, name)

def ortho_weight(ndim, rng=None):
    '''Make ortho weight tensor.'''
    if not rng:
        rng = rng_
    W = rng.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype('float32')

def norm_weight(nin, nout=None, scale=0.01, ortho=True, rng=None):
    '''Make normal weight tensor.'''
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
    '''Estimate parzen window.'''
    log_p = 0.
    d = samples.shape[-1]
    z = d * np.log(h * np.sqrt(2 * np.pi))
    for test in tests:
        d_s = (samples - test[None, :]) / h
        e = log_mean_exp((-.5 * d_s ** 2).sum(axis=1), as_numpy=True, axis=0)
        log_p += e - z
    return log_p / float(tests.shape[0])

def get_w_tilde(log_factor):
    '''Gets normalized weights'''
    log_factor = log_factor - T.log(log_factor.shape[0]).astype(floatX)
    w_norm   = log_sum_exp(log_factor, axis=0)
    log_w    = log_factor - T.shape_padleft(w_norm)
    w_tilde  = T.exp(log_w)
    return w_tilde

def log_mean_exp(x, axis=None, as_numpy=False):
    '''Numerically stable log(exp(x).mean())'''
    if as_numpy:
        Te = np
    else:
        Te = T
    x_max = Te.max(x, axis=axis, keepdims=True)
    return Te.log(Te.mean(Te.exp(x - x_max), axis=axis, keepdims=True)) + x_max

def log_sum_exp(x, axis=None):
    '''Numerically stable log( sum( exp(A) ) ).'''
    x_max = T.max(x, axis=axis, keepdims=True)
    y = T.log(T.sum(T.exp(x - x_max), axis=axis, keepdims=True)) + x_max
    y = T.sum(y, axis=axis)
    return y

def concatenate(tensor_list, axis=0):
    '''
    Alternative implementation of `theano.T.concatenate`.
    This function does exactly the same thing, but contrary to Theano's own
    implementation, the gradient is implemented on the GPU.
    Backpropagating through `theano.tensor.concatenate` yields slowdowns
    because the inverse operation (splitting) needs to be done on the CPU.
    This implementation does not have that problem.

    Usage:
        >>> x, y = theano.tensor.matrices('x', 'y')
        >>> c = concatenate([x, y], axis=1)
    Args:
        tensor_list: list, list of Theano tensor expressions that should be concatenated.
        axis: int, the tensors will be joined along this axis.
    Returns:
        out: tensor, the concatenated tensor expression.

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
