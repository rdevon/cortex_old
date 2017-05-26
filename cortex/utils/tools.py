'''
Helper module for learning.
'''

from collections import OrderedDict
from ConfigParser import ConfigParser
import importlib
import logging
import numpy as np
import os
import pprint
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
profile = False

# For getting terminal column width
try:
    _, _columns = os.popen('stty size', 'r').read().split()
    _columns = int(_columns)
except ValueError:
    _columns = 1

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

def get_subclasses(module):
    '''Resolves the subclass from str.

    '''
    resolve_dict = dict()

    try:
        classes = module._classes
    except AttributeError:
        classes = []

    for c in classes:
        resolve_dict[c] = getattr(module, c)

    try:
        submodules = module._modules
    except AttributeError:
        submodules = []

    for submodule in submodules:
        submodule = importlib.import_module(module.__name__ + '.' + submodule)
        resolve_dict.update(**resolve_subclasses(submodule))

    return resolve_dict


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
                pass
            try:
                param_key = model.name + '_' + k
                pretrained_v = pretrained_kwargs.pop(param_key)
                logger.info('Found %s for %s %s'
                            % (k, model.name, pretrained_v.shape))
                assert model.params[k].shape == pretrained_v.shape, (
                    'Sizes do not match for model %s, parameter %s: %s vs %s'
                    % (model.name, k, model.params[k].shape, pretrained_v.shape)
                )
                model.params[k] = pretrained_v
                warnings.warn('Old style parameter naming found',
                              FutureWarning)
            except KeyError:
                pass
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

def check_bad_nums(rvals, logger=None):
    '''Checks for nans and infs.

    Args:
        rvals (dict)

    Returns:
        bool

    '''
    found = False
    for k, v in rvals.iteritems():
        if np.any(np.isnan(v)):
            if logger is not None: logger.error('Found nan num (%s)' % k)
            found = True
        elif np.any(np.isinf(v)):
            if logger is not None: logger.error('Found inf (%s)' % k)
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
    if pp is None:
        return name
    return '%s.%s' % (pp, name)
