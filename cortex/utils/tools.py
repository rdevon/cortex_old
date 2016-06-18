'''
Helper module for learning.
'''

from collections import OrderedDict
from ConfigParser import ConfigParser
import logging
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

from . import floatX
from . import logger


logger = logging.getLogger(__name__)
random_seed = random.randint(1, 10000)
rng_ = np.random.RandomState(random_seed)
profile = False

# For getting terminal column width
_, _columns = os.popen('stty size', 'r').read().split()
_columns = int(_columns)

def print_section(s):
    '''For printing sections to scripts nicely.

    Args:
        s (str): string of section

    '''
    head = '[CORTEX]:'
    h = head + s + ('-' * (_columns - len(head) - len(s)))
    print h

def get_paths():
    '''Pulls all paths from `~/.cortexrc` file.

    Returns:
        dict: dictionary of paths from `~.cortexrc`.

    '''
    d = os.path.expanduser('~')
    config_file = os.path.join(d, '.cortexrc')
    config = ConfigParser()
    config.read(config_file)

    try:
        path_dict = config._sections['PATHS']
    except KeyError as e:
        raise ValueError('There is a problem with the .cortexrc file. Either '
                         'create this file in your $HOME directory, or run '
                         '`cortex-setup`. If this file already exists, please '
                         'make sure that the `[PATHS]` field exists.')

    path_dict.pop('__name__')
    return path_dict

def resolve_path(p):
    '''Resolves a path using the `.cortexrc` file.

    Args:
        p (str): path string.

    Returns:
        str: path with substrings replaced from `~/.cortexrc`.

    '''
    path_dict = get_paths()
    for k, v in path_dict.iteritems():
        p = p.replace(k, v)
    pieces = p.split('/')
    for piece in pieces:
        if piece.startswith('$'):
            raise ValueError('Field in .cortexrc file is missing and may need '
                             'to be filled: %s' % piece)
    return p

def get_srng():
    '''Shared Randomstream.

    '''
    srng = SRandomStreams(random.randint(1, 1000000))
    return srng

def get_trng():
    '''Normal Randomstream.

    '''
    trng = RandomStreams(random.randint(1, 1000000))
    return trng

def warn_kwargs(c, kwargs):
    '''Warns of extra keyword arguments.

    Args:
        c (object): class
        **kwargs: extra keyword arguments.

    '''
    if len(kwargs) > 0:
        logger.warn('Class instance %s has leftover kwargs %s'
                    % (type(c), kwargs))

def update_dict_of_lists(d_to_update, **d):
    '''Updates a dict of list with kwargs.

    Args:
        d_to_update (dict): dictionary of lists.
        **d: keyword arguments to append.

    '''
    for k, v in d.iteritems():
        if k in d_to_update.keys():
            d_to_update[k].append(v)
        else:
            d_to_update[k] = [v]

def debug_shape(X, x, t_out, updates=None):
    '''Debug function that returns the shape then raises assert error.

    Raises assert False. For debugging shape only.

    Args:
        X (T.tensor): input tensor.
        x (numpy.array): input values.
        t_out (T.tensor): output tensor.
        updates (theano.OrderedUpdates): updates for function.

    '''
    f = theano.function([X], t_out, updates=updates)
    out = f(x)
    print out.shape
    assert False

def print_profile(tparams):
    '''Prints shapes of the shared variables.

    Args:
        tparams (dict): parameters to print shape of.

    '''
    s = 'Printing profile for tparams (name, shape): '
    for (k, v) in tparams.iteritems():
        s += '\n\t%s %s' % (k, v.get_value().shape)
    logger.info(s)

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
        profile=profile,
        strict=strict
    )

def init_weights(model, weight_noise=False, weight_scale=0.001, dropout=False,
                 **kwargs):
    '''Inialization function for weights.

    Args:
        model (Layer).
        weight_noise (bool): noise the weights.
        weight_scale (float): scale for weight initialization.
        dropout (bool): use dropout.
        **kwargs: extra kwargs.

    Returns:
        dict: extra kwargs.

    '''
    model.weight_noise = weight_noise
    model.weight_scale = weight_scale
    model.dropout = dropout
    return kwargs

def init_rngs(model, rng=None, trng=None, **kwargs):
    '''Initialization function for RNGs.

    Args:
        model (Layer).
        rng (np.randomStreams).
        trng (theano.randomStreams).
        **kwargs: extra kwargs.

    Returns:
        dict: extra kwargs.

    '''
    if rng is None:
        rng = rng_
    model.rng = rng
    if trng is None:
        model.trng = RandomStreams(random.randint(1, 10000))
    else:
        model.trng = trng
    return kwargs

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

def load_experiment(experiment_yaml):
    '''Load an experiment from a yaml.

    Args:
        experiment_yaml (str): path to yaml

    Returns:
        dict: extracted yaml.

    '''
    logger.info('Loading experiment from %s' % experiment_yaml)
    exp_dict = yaml.load(open(experiment_yaml))
    logger.debug('Experiment hyperparams: \n%s' % pprint.pformat(exp_dict))
    return exp_dict

def load_model(model_file, f_unpack=None, strict=True, **extra_args):
    '''Loads pretrained model.

    Args:
        model_file (str): path to file.
        f_unpack (function): unpacking function.
            Must return tuple of::

                models: a list of Layer objects
                pretrained_kwargs: a dictionary of saved parameters.
                kwargs: dictionary of extra arguments (can be None).

            See `cortex.models.rnn.unpack` for an example.
        strict (bool): fail on extra parameters.
        **extra_args: extra keyword arguments to pass to unpack.

    Returns:
        dict: dictionary of models.
        dict: extra keyword arguments.

    '''

    logger.info('Loading model from %s' % model_file)
    params = np.load(model_file)
    d = dict()
    for k in params.keys():
        try:
            d[k] = params[k].item()
        except ValueError:
            d[k] = params[k]

    d.update(**extra_args)
    models, pretrained_kwargs, kwargs = f_unpack(**d)

    logger.info('Pretrained model(s) has the following parameters: \n%s'
          % pprint.pformat(pretrained_kwargs.keys()))

    model_dict = OrderedDict()

    for model in models:
        if model is None:
            continue
        logger.info('--- Loading params for %s' % model.name)
        for k, v in model.params.iteritems():
            try:
                param_key = _p(model.name, k)
                pretrained_v = pretrained_kwargs.pop(param_key)
                logger.info('Found %s for %s %s'
                            % (k, model.name, pretrained_v.shape))
                assert model.params[k].shape == pretrained_v.shape, (
                    'Sizes do not match for model %s, parameter %s: %s vs %s'
                    % (model.name, k, model.params[k].shape, pretrained_v.shape)
                )
                model.params[k] = pretrained_v
            except KeyError:
                try:
                    param_key = '{key}'.format(key=k)
                    pretrained_v = pretrained_kwargs[param_key]
                    logger.info('Found %s, but name is ambiguous' % k)
                    assert model.params[k].shape == pretrained_v.shape, (
                        'Sizes do not match: %s vs %sfor model %s, parameter '
                        '%s: %s vs %s'
                        % (model.name, k, model.params[k].shape,
                           pretrained_v.shape)
                    )
                    model.params[k] = pretrained_v
                except KeyError:
                    logger.info('{} not found'.format(k))
        model_dict[model.name] = model

    if len(pretrained_kwargs) > 0 and strict:
        raise ValueError('ERROR: Leftover params: %s' %
                         pprint.pformat(pretrained_kwargs.keys()))
    elif len(pretrained_kwargs) > 0:
        logger.warn('Leftover params: %s' %
                      pprint.pformat(pretrained_kwargs.keys()))

    return model_dict, kwargs

def check_bad_nums(rvals):
    '''Checks for nans and infs.

    Args:
        rvals (dict)

    Returns:
        bool

    '''
    found = False
    for k, v in rvals.iteritems():
        if np.any(np.isnan(v)):
            logger.error('Found nan num (%s)' % k)
            found = True
        elif np.any(np.isinf(v)):
            logger.error('Found inf (%s)' % k)
            found = True
    return found

def flatten_dict(d):
    '''Flattens a dictionary of dictionaries.

    '''
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
    return '%s.%s'%(pp, name)

def ortho_weight(ndim, rng=None):
    '''Make ortho weight tensor.

    '''
    if not rng:
        rng = rng_
    W = rng.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype(floatX)

def norm_weight(nin, nout=None, scale=0.01, ortho=True, rng=None):
    '''Make normal weight tensor.

    '''
    if not rng:
        rng = rng_
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin, rng=rng)
    else:
        W = scale * rng.randn(nin, nout)
    return W.astype(floatX)

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

def get_w_tilde(log_factor):
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
    if as_numpy:
        Te = np
    else:
        Te = T
    x_max = Te.max(x, axis=axis, keepdims=True)
    return Te.log(Te.mean(Te.exp(x - x_max), axis=axis, keepdims=True)) + x_max

def log_sum_exp(x, axis=None):
    '''Numerically stable log( sum( exp(A) ) ).

    '''
    x_max = T.max(x, axis=axis, keepdims=True)
    y = T.log(T.sum(T.exp(x - x_max), axis=axis, keepdims=True)) + x_max
    y = T.sum(y, axis=axis)
    return y

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

    from math import sqrt, floor

    n = floor(sqrt(x))
    while True:
        if n < 1:
            raise ValueError
        rem = x % n
        if rem == 0:
            return int(n), int(x / n)
        n -= 1

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

    Argsuments:
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
