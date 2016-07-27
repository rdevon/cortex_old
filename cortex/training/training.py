'''
Generic training scripts.

These scripts are meant as a basic example for building scripts for training,
not as a basis for all training.

'''

import sys

import argparse
from collections import OrderedDict
from glob import glob
import logging
if not 'matplotlib' in sys.modules:
    import matplotlib
    matplotlib.use('Agg')
import numpy as np
import os
from os import path
from progressbar import (
    Bar,
    Percentage,
    ProgressBar,
    RotatingMarker,
    SimpleProgress,
    Timer
)
import shutil
import theano
from theano import tensor as T
import time

from ..utils import logger as cortex_logger
from . import op
from ..utils.learning_scheduler import Scheduler
from ..utils.tools import (
    check_bad_nums,
    itemlist,
    load_experiment,
    load_model,
    resolve_path,
    update_dict_of_lists,
    warn_kwargs
)


np.set_printoptions(threshold=np.nan)
logger = logging.getLogger(__name__)

def set_experiment(args):
    '''Generic experiment setup method.

    Extracts args from a parser like from `make_argument_parser`. These are
    extracted into a dictionary for kwargs.

    Args:
        args (dict or argparse.args).

    Returns:
        dict: dictionary of experiment arguments.

    '''
    try:
        args = vars(args)
    except TypeError:
        pass

    verbosity = args.pop('verbosity', 1)
    cortex_logger.set_stream_logger(verbosity)

    if 'load_model' in args.keys():
        load_model = args.pop('load_model')
    else:
        load_model = None

    if 'load_last' in args.keys():
        load_last = args.pop('load_last')
    else:
        load_last = False

    args = OrderedDict((k, v) for k, v in args.iteritems() if v is not None)

    try:
        exp_dict = load_experiment(path.abspath(args['experiment']))
    except KeyError:
        logger.info('No experiment yaml found. Using defaults.')
        exp_dict = dict()
    exp_dict.update(args)

    if not exp_dict.get('out_path', False):
        exp_dict['out_path'] = resolve_path('$outs')

    if not exp_dict.get('name', False):
        exp_dict['name'] = '.'.join(
            exp_dict['module'].split('/')[-1].split('.')[:-1])

    exp_dict['out_path'] = path.join(exp_dict['out_path'], exp_dict['name'])
    out_path = exp_dict['out_path']

    if path.isfile(out_path):
        raise ValueError()
    elif not path.isdir(out_path):
        os.mkdir(path.abspath(out_path))

    cortex_logger.set_file_logger(path.join(out_path, exp_dict['name'] + '.log'))
    logging.info('Starting experiment %s. Saving to %s'
                 % (exp_dict['name'], out_path))

    if exp_dict.get('experiment', False):
        experiment = exp_dict.pop('experiment')
        shutil.copy(
            path.abspath(experiment), path.abspath(out_path))

    if load_model is not None:
        model_to_load = load_model
    elif load_last:
        model_to_load = glob(path.join(out_path, '*last.npz'))[0]
    else:
        model_to_load = None

    exp_dict['model_to_load'] = model_to_load
    return exp_dict

def reload_model(args):
    '''Reloads a model from argparse args.

    Extracts model into a dictionary from argparse args like from
    `make_argument_parser_test`.

    Args:
        args (argparse.args).

    Returns:
        dict: dictionary of experiment arguments.

    '''
    exp_dir = path.abspath(args.experiment_dir)
    out_path = path.join(exp_dir, 'results')
    if not path.isdir(out_path):
        os.mkdir(out_path)

    try:
        yaml = glob(path.join(exp_dir, '*.yaml'))[0]
        logging.info('Found yaml %s' % yaml)
    except:
        raise ValueError('yaml file not found. Cannot reload experiment.')

    exp_dict = load_experiment(path.abspath(yaml))
    exp_dict['out_path'] = out_path

    try:
        if args.best:
            tag = 'best'
        else:
            tag = 'last'
        model_file = glob(path.join(exp_dir, '*%s*npz' % tag))[0]
        logging.info('Found %s in %s' % (tag, model_file))
    except:
        raise ValueError()

    params = np.load(model_file)
    try:
        logging.info('Loading dataset arguments from saved model.')
        dataset_args = params['dataset_args'].item()
        exp_dict.update(dataset_args=dataset_args)
    except KeyError:
        pass

    exp_dict['model_to_load'] = model_file
    args = vars(args)
    args = OrderedDict((k, v) for k, v in args.iteritems() if v is not None)
    exp_dict.update(**args)
    return exp_dict

def set_model(create_model, model_to_load, unpack, **kwargs):
    '''Convenience method for creating new or loading old model.

    Object creation often can be reduced to 3 things: a creation method,
    a saved model, and a means of unpacking the saved model. This method
    attempts to summarize these into one method.

    Args:
        create_model (function): method for creating model.
            No arguments. Methods should be defined out-of-method, e.g.::

                dim = 100
                def create_model():
                    return ModelClass(dim)

            and then passed into `set_model`
        model_to_load (str): path the npz file.
        unpack (function): Takes model_to_load.
            See `utils.tools.load_model` for details.

    Returns:
        dict: dictionary of Layer subclass objects

    '''
    if model_to_load is not None:
        models, _ = load_model(model_to_load, unpack, **kwargs)
    else:
        models = create_model()
    return models
