'''Trainer class.

Does typical training work.

'''

from collections import OrderedDict
import imp
import inspect
import logging
import numpy as np
from os import path
import pprint
import sys
import theano
from theano import tensor as T
import types

from ..utils import floatX, intX
from ..models import Layer
from ..utils.monitor import SimpleMonitor
from ..utils.preprocessor import Preprocessor
from ..utils.tools import _p, load_model as _load_model, print_profile, print_section
from .training import (
    main_loop, make_argument_parser_trainer, set_experiment, set_model,
    set_optimizer, set_params)
from ..datasets import build_datasets


logger = logging.getLogger('cortex')

def setup(module):
    '''Sets up module.

    '''
    if hasattr(module, 'setup'):
        print_section('Running setup')
        module.setup()

def set_data(module):
    '''Sets the datasets.

    '''
    print_section('Setting up data')
    if hasattr(module, 'set_data'):
        datasets = module.data()
    else:
        datasets = dataset_factory(module.resolve_dataset, **module.dataset_args)
    if 'idx' in datasets.keys():
        module.dataset_args['idx'] = datasets['idx']

    logger.debug('Formed datasets %s of type %s. '
                 'Data distributions are %s, '
                 'dimensions are %s'
                 % (datasets.keys(),
                    datasets['train'].__class__,
                    datasets['train'].distributions,
                    datasets['train'].dims))

    module.dataset = datasets['train']
    module.valid_dataset = datasets['valid']
    module.test_dataset = datasets['test']


def build(module, model_to_load=None):
    '''Forms the models.

    '''
    if module.dataset is None:
        raise ValueError('Module data must be set before forming models.')

    if model_to_load is not None:
        models, _ = load_model(model_to_load, module.unpack, **kwargs)
    else:
        models = module.build()
    module.models = models
    return set_tparams(module)

def set_tparams(module):
    tparams = OrderedDict()
    for k, v in module.models.iteritems():
        tparams.update(**v.set_tparams())
    module.tparams = tparams
    return tparams

def set_cost(module):
    '''Sets costs.

    '''

    used_inputs, results, updates, constants, outputs = module.cost()
    if isinstance(updates, list):
        updates = theano.OrderedUpdates(updates)
    logger.debug('Model has the following '
                 '\n\tresult keys: %s'
                 '\n\tconstants: %s'
                 '\n\tupdates: %s'
                 % (results.keys(), constants, updates.keys()))
    inputs = OrderedDict((k, v) for k, v in module.inputs.iteritems()
        if k in used_inputs)
    module.input_keys = used_inputs
    module.inputs = inputs
    return results, updates, constants, outputs

def set_test_function(module, results, outputs):
    '''Sets the test function of a module.

    '''
    if hasattr(module, 'test'):
        f_test = module.test(results, outputs)
    else:
        f_test = theano.function(module.inputs.values(), results)
    return f_test

def set_out_function(module, results, outputs):
    '''Sets function for outputs.
    '''
    outs = OrderedDict()
    for k, v in outputs.iteritems():
        if isinstance(v, list):
            for i, _v in enumerate(v):
                outs['%s_%d' % (k, i)] = _v
        else:
            outs[k] = v
    f_outs = theano.function(module.inputs.values(), outs)
    return f_outs

def set_save_function(module, tparams):
    '''Sets the save function of a module.

    '''
    def save(outfile):
        d = dict((k, v.get_value()) for k, v in tparams.items())
        d.update(**module.args)
        d.update(**dict((k, module.__dict__[k]) for k in module._save_fields))
        np.savez(outfile, **d)

    if hasattr(module, 'save'):
        f_save = module.save
    else:
        f_save = save
    return f_save

def set_viz_function(module, results, outputs):
    '''Sets the visualization function of a module.

    '''
    if hasattr(module, 'viz'):
        print_section('Setting visualization function.')
        f_viz = module.viz(results, outputs)
        logger.info('Testing visualization function.')
        f_viz()
        logger.info('Done testing visualization.')
        return f_viz
    else:
        return None

def set_eval_functions(module, **kwargs):
    if hasattr(module, 'eval'):
        return module.eval(**kwargs)
    else:
        return OrderedDict()

def set_profile_function(tparams):
    results = OrderedDict()
    for k, v in tparams.iteritems():
        results[k + '_mean'] = v.mean()
        results[k + '_min'] = v.min()
        results[k + '_max'] = v.max()

    return theano.function([], results)

def check(module):
    '''Runs checks.

    '''
    if hasattr(module, 'check'):
        logger.info('Checking experiment.')
        module.check()

def finish(module):
    '''Extra finishing-up.

    '''
    if hasattr(module, 'finish'):
        logger.info('Finishing up setup')
        module.finish()

def train(module, cost, tparams, updates, constants, f_test=None, f_save=None,
          f_viz=None, f_outs=None, f_profile=None, test_every=10, show_every=10,
          monitor_gradients=False):
    print_section('Getting gradients and building optimizer.')

    excludes = module.learning_args.pop('excludes', [])
    tparams, all_params = set_params(tparams, updates, excludes=excludes)
    f_grad_shared, f_grad_updates, learning_args = set_optimizer(
        module.inputs.values(), cost, tparams, constants, updates, [],
        **module.learning_args)

    print_section('Actually running (main loop)')
    monitor = SimpleMonitor()

    main_loop(
        module.dataset, module.valid_dataset,
        f_grad_shared, f_grad_updates, f_test,
        save=f_save, save_images=f_viz, f_outs=f_outs, f_profile=f_profile,
        monitor=monitor, monitor_gradients=monitor_gradients,
        out_path=module.out_path,
        name=module.name,
        test_every=test_every,
        show_every=show_every,
        input_keys=module.input_keys,
        **learning_args)


class Trainer(object):
    '''Trainer class for cortex.

    Trainer processes and organizes scripts for use in cortex.

    '''

    def run(self, module, model_to_load=None, **kwargs):
        setup(module)
        set_data(module)
        make_inputs(module)
        tparams = build(module, model_to_load=model_to_load)
        print_profile(tparams)
        results, updates, constants, outputs = set_cost(module)
        cost = results['cost']

        f_test = set_test_function(module, results, outputs)
        f_save = set_save_function(module, tparams)
        f_viz = set_viz_function(module, results, outputs)
        f_outs = set_out_function(module, results, outputs)
        f_profile = set_profile_function(tparams)

        check(module)
        finish(module)

        kwargs.update(
            f_test=f_test,
            f_save=f_save,
            f_viz=f_viz,
            f_outs=f_outs,
            f_profile=f_profile
        )

        train(module, cost, tparams, updates, constants, **kwargs)


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = make_argument_parser_trainer()
    args = parser.parse_args()
    exp_dict = set_experiment(args)
    module = ModuleContainer(
        path.abspath(exp_dict.pop('module')),
        preprocessing=exp_dict.pop('preprocessing', None),
        name=exp_dict.pop('name', None),
        out_path=exp_dict.pop('out_path', None))

    module.update(exp_dict)
    show_every = exp_dict.pop('show_every', 10)
    test_every = exp_dict.pop('test_every', 10)
    monitor_gradients = exp_dict.pop('monitor_gradients', False)
    model_to_load = exp_dict.pop('model_to_load', None)

    trainer = Trainer()
    trainer.run(module, show_every=show_every, test_every=test_every,
                model_to_load=model_to_load, monitor_gradients=monitor_gradients)