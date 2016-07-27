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

def make_argument_parser():
    '''Generic experiment parser.

    Generic parser takes the experiment yaml as the main argument, but has some
    options for reloading, etc. This parser can be easily extended using a
    wrapper method.

    Returns:
        argparse.parser

    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', default=None)
    parser.add_argument('-o', '--out_path', default=None,
                        help='Output path for stuff')
    parser.add_argument('-r', '--load_last', action='store_true')
    parser.add_argument('-l', '--load_model', default=None)
    parser.add_argument('-n', '--name', default=None)
    parser.add_argument('-v', '--verbosity', type=int, default=1,
                        help='Verbosity of the logging. (0, 1, 2)')
    return parser

def make_argument_parser_trainer():
    '''Generic experiment parser for a trainer.

    Generic parser takes the experiment yaml as the main argument, but has some
    options for reloading, etc. This parser can be easily extended using a
    wrapper method.

    Returns:
        argparse.parser

    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('module', default=None)
    parser.add_argument('experiment', nargs='?', default=None)
    parser.add_argument('-o', '--out_path', default=None,
                        help='Output path for stuff')
    parser.add_argument('-r', '--load_last', action='store_true')
    parser.add_argument('-l', '--load_model', default=None)
    parser.add_argument('-n', '--name', default=None)
    parser.add_argument('-g', '--monitor_gradients', action='store_true')
    parser.add_argument('-v', '--verbosity', type=int, default=1,
                        help='Verbosity of the logging. (0, 1, 2)')
    return parser

def make_argument_parser_test():
    '''Generic experiment parser for testing.

    Takes the experiment directory as the argument in command line.

    Returns:
        argparse.parser

    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_dir', default=None)
    parser.add_argument('-m', '--mode', default='valid',
                        help='Dataset mode: valid, test, or train')
    parser.add_argument('-b', '--best', action='store_true',
                        help='Load best instead of last saved model.')
    parser.add_argument('-v', '--verbosity', type=int, default=1,
                        help='Verbosity of the logging. (0, 1, 2)')
    return parser

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

def set_tparams(model_dict):
    '''Generic tparams setter.

    Args:
        model_dict (dict)

    '''
    tparams = OrderedDict()
    for model in model_dict.values():
        tparams.update(**model.set_tparams())
    return tparams

def set_params(tparams, updates, excludes=[]):
    '''Sets params, removing updates from tparams.

    Convenience function to extract the theano parameters that will have
    gradients calculated.

    Args:
        tparams (dict): dictionary of Theano shared variables.
        updates (theano.OrderedUpdates): used to exclude variables that have
            gradients calculated.
        excludes (list): list of keys to exclude from gradients or learning.

    Returns:
        OrderedDict: dict of variables that will have gradients calculated.
        OrderedDict: dict of variables that will be saved.

    '''
    all_params = OrderedDict((k, v) for k, v in tparams.iteritems())

    tparams = OrderedDict((k, v)
        for k, v in tparams.iteritems()
        if (v not in updates.keys()) and (k not in excludes))

    logging.info('Learned model params: %s' % tparams.keys())
    logging.info('Saved params: %s' % all_params.keys())

    return tparams, all_params

def test(data_iter, f_test, f_test_keys, input_keys, n_samples=None):
    '''Tests the model using a data iterator.

    Args:
        data_iter (Dataset): dataset iterator.
        f_test (theano.function)
        f_test_keys (list): The keys that go with the corresponding list
            of outputs from `f_test`.
        input_keys (list): Used to extract multiple modes from dataset
            for `f_test`.
        n_samples (Optional[int]) If not None, use only this number of samples
            as input to `f_test`.

    Returns:
        OrderedDict: dictionary of np.array results.

    '''
    data_iter.reset()
    maxvalid = data_iter.n

    widgets = ['Testing (%s set): ' % data_iter.mode, Percentage(),
               ' (', Timer(), ')']
    pbar    = ProgressBar(widgets=widgets, maxval=maxvalid).start()
    results = OrderedDict()
    while True:
        try:
            outs = data_iter.next()
            inps = [outs[k] for k in input_keys]
            r = f_test(*inps)
            if isinstance(r, dict):
                results_i = r
            else:
                results_i = dict((k, v) for k, v in zip(f_test_keys, r))

            for k, v in results_i.iteritems():
                if isinstance(v, theano.sandbox.cuda.CudaNdarray):
                    results_i[k] = np.asarray(v)

            update_dict_of_lists(results, **results_i)

            if data_iter.pos == -1:
                pbar.update(maxvalid)
            else:
                pbar.update(data_iter.pos)

        except StopIteration:
            print
            break

    for k, v in results.iteritems():
        try:
            results[k] = np.mean(v)
        except Exception as e:
            logging.error(k)
            logging.error(v)
            raise e

    data_iter.reset()

    return results

def validate(results, best_valid, e, best_epoch, save=None, valid_key=None,
             valid_sign=None, bestfile=None, **kwargs):
    '''Generic validation method.

    Compares the validation result against previous best.

    Args:
        results (OrderedDict): dictionary of np.array results.
        best_valid (float): Best pervious value.
        e (int): Epoch
        best_epoch (int): Epoch for best_valid.
        save (function): Method for saving params.
        valid_key (str): Key from results to test against best_valid.
        bestfile (str): Path to best file.

    Returns:
        float: best valid
        int: best epoch

    '''
    warn_kwargs(None, kwargs)

    valid_value = results[valid_key]
    if valid_sign == '-':
        valid_value *= -1

    if valid_value < best_valid:
        print 'Found best %s: %.2f' % (valid_key, valid_value)
        best_valid = valid_value
        best_epoch = e
        if save is not None and bestfile is not None:
            print 'Saving best to %s' % bestfile
            save(bestfile)
    else:
        print 'Best (%.2f) at epoch %d' % (best_valid, best_epoch)

    return best_valid, best_epoch
