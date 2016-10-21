'''Setup script

'''

from collections import OrderedDict
from glob import glob
import logging
import os
from os import path
import pprint
import readline, glob
import sys
import urllib2
import yaml

if not 'matplotlib' in sys.modules:
    import matplotlib
    matplotlib.use('Agg')
import numpy as np
from progressbar import (
    Bar,
    Percentage,
    ProgressBar,
    RotatingMarker,
    SimpleProgress,
    Timer
)
import shutil
import time


from .utils import logger as cortex_logger
from .utils.tools import _p, get_paths, resolve_path
from .utils.extra import (
    complete_path, query_yes_no, write_default_theanorc, write_path_conf)


np.set_printoptions(threshold=np.nan)
logger = logging.getLogger(__name__)

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
    autoname = args.pop('autoname')
    cortex_logger.set_stream_logger(verbosity)

    if 'load_model' in args.keys():
        load_model = args.pop('load_model')
    else:
        load_model = None

    if 'load_last' in args.keys():
        load_last = args.pop('load_last')
    else:
        load_last = False

    args = OrderedDict((k, v) for k, v in args.items() if v is not None)
    if autoname:
        name = args.pop('name', 'model')
        for k, v in args.items():
            if k in ['source', 'test']: continue
            tk = ''.join([k_[0] for k_ in k.split('_')])
            name += '.{key}={value}'.format(key=tk, value=v)
        args['name'] = name

    '''
    try:
        exp_dict = load_experiment(path.abspath(args['experiment']))
    except KeyError:
        logger.info('No experiment yaml found. Using defaults.')
    '''
    exp_dict = dict()

    exp_dict.update(args)

    #cortex_logger.set_file_logger(path.join(out_path, exp_dict['name'] + '.log'))
    #logging.info('Starting experiment %s. Saving to %s'
    #             % (exp_dict['name'], out_path))

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

    if model_to_load is not None: cortex.load(model_to_load)
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
    args = OrderedDict((k, v) for k, v in args.items() if v is not None)
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

def main():
    from .datasets import fetch_basic_data
    from .datasets.neuroimaging import fetch_neuroimaging_data

    readline.set_completer_delims(' \t\n;')
    readline.parse_and_bind('tab: complete')
    readline.set_completer(complete_path)
    print ('Welcome to Cortex: a deep learning toolbox for '
            'neuroimaging')
    print ('Cortex requires that you enter some paths for '
            'default dataset and output directories. These '
            'can be changed at any time and are customizable '
            'via the ~/.cortexrc file.')

    try:
        path_dict = get_paths()
    except ValueError:
        path_dict = dict()

    if '$data' in path_dict:
        data_path = raw_input(
            'Default data path: [%s] ' % path_dict['$data']) or path_dict['$data']
    else:
        data_path = raw_input('Default data path: ')
    data_path = path.expanduser(data_path)
    if not path.isdir(data_path):
        raise ValueError('path %s does not exist. Please create it.' % data_path)

    if '$outs' in path_dict:
        out_path = raw_input(
            'Default output path: [%s] ' % path_dict['$outs']) or path_dict['$outs']
    else:
        out_path = raw_input('Default output path: ')
    out_path = path.expanduser(out_path)
    if not path.isdir(out_path):
        raise ValueError('path %s does not exist. Please create it.' % out_path)
    write_path_conf(data_path, out_path)

    print ('Cortex demos require additional data that is not necessary for '
           'general use of the Cortex as a package.'
           'This includes MNIST, Caltech Silhoettes, and some UCI dataset '
           'samples.')

    answer = query_yes_no('Download basic dataset? ')

    if answer:
        try:
            fetch_basic_data()
        except urllib2.HTTPError:
            print 'Error: basic dataset not found.'

    print ('Cortex also requires neuroimaging data for the neuroimaging data '
           'for the neuroimaging demos. These are large and can be skipped.')

    answer = query_yes_no('Download neuroimaging dataset? ')

    if answer:
        try:
            fetch_neuroimaging_data()
        except urllib2.HTTPError:
            print 'Error: neuroimaging dataset not found.'

    home = path.expanduser('~')
    trc = path.join(home, '.theanorc')
    if not path.isfile(trc):
        print 'No %s found, adding' % trc
        write_default_theanorc()
