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

from . import floatX, intX
from .monitor import SimpleMonitor
from .preprocessor import Preprocessor
from .tools import print_section
from .training import (
    main_loop, make_argument_parser_trainer, set_experiment, set_model,
    set_optimizer)
from ..datasets import dataset_factory


logger = logging.getLogger('cortex')


class Trainer(object):
    '''Trainer class for cortex.

    Trainer processs and organizes scripts for use in cortex.

    '''
    __training_args = ['test_every', 'show_every']

    def __init__(self, module, preprocessing=None, out_path=None):
        print_section('Setting up trainer')
        module.preprocessor = Preprocessor(preprocessing)
        self.out_path = out_path

    def setup(self, module):
        if hasattr(module, 'setup'):
            print_section('Running setup')
            module.setup(module)

    def set_data(self, module):
        print_section('Setting up data')
        if hasattr(module, 'set_data'):
            data = module.set_data(module)
        else:
            data = dataset_factory(module.resolve_dataset,
                                   **module.args['dataset'])
        if 'idx' in data.keys():
            module.args['dataset']['idx'] = data['idx']

        return data

    def make_inputs(self, module, data):
        data = data['train']

        print_section('Setting inputs')
        d = data.next()
        inps = OrderedDict()
        for k, v in d.iteritems():
            if v.ndim == 1:
                C = T.vector
            elif v.ndim == 2:
                C = T.matrix
            elif v.ndim == 3:
                C = T.tensor3
            else:
                raise ValueError('Data dim over 3 not supported.')

            if v.dtype == floatX:
                dtype = floatX
            elif v.dtype == intX:
                dtype = intX
            else:
                raise ValueError('dtype %s not supported' % v.dtype)

            X = T.tensor3(k, dtype=dtype)
            inps[k] = X
        logger.debug('Dataset has the following inputs: %s' % inps)
        data.reset()
        return inps

    def form_models(self, module, data, model_to_load=None):
        if model_to_load is not None:
            models, _ = load_model(model_to_load, module.unpack, **kwargs)
        else:
            models = module.create_model(module, data)

        tparams = OrderedDict()
        for k, v in models.iteritems():
            tparams.update(**v.set_tparams())
        return models, tparams

    def set_cost(self, module, models, inputs):
        used_inputs, results, updates, constants, model_outs = module.set_cost(
            module, models, inputs)
        logger.debug('Model has the following '
                     '\n\tresult keys: %s'
                     '\n\tconstants: %s'
                     '\n\tupdates: %s'
                     % (results.keys(), updates.keys(), constants))
        inputs = OrderedDict((k, v) for k, v in inputs.iteritems()
            if k in used_inputs)
        return inputs, results, updates, constants, model_outs

    def set_test_function(self, module, inputs, results, model_outs, updates):
        if hasattr(module, 'set_test_function'):
            print_section('Setting test function')
            f_test = module.set_test_function(
                module, inputs, results, model_outs, updates)
            return f_test
        else:
            return None

    def set_save_function(self, module, tparams):
        def save(outfile):
            d = dict((k, v.get_value()) for k, v in tparams.items())
            d.update(**dict((k + '_args', v) for k, v in module.args.iteritems()))
            np.savez(outfile, **d)

        if hasattr(module, 'set_save_function'):
            f_save = module.set_save_function(module)
        else:
            f_save = save
        return f_save

    def set_analysis_function(self, module, inputs, data, results, models,
                              model_outs, updates):
        if hasattr(module, 'set_analysis_function'):
            print_section('Setting analysis function')
            f_anal = module.set_analysis_function(
                module, inputs, data, results, models, model_outs, updates,
                self.out_path)
            return f_anal
        else:
            return None

    def train(self, module, inputs, cost, tparams, updates, constants, data,
              f_test=None, f_save=None, f_analysis=None, test_every=None,
              show_every=None):
        print_section('Getting gradients and building optimizer.')

        inputs = inputs.values()
        f_grad_shared, f_grad_updates, learning_args = set_optimizer(
            inputs, cost, tparams, constants, updates, [],
            **module.args['learning'])

        print_section('Actually running (main loop)')
        monitor = SimpleMonitor()

        main_loop(
            data['train'], data['valid'],
            f_grad_shared, f_grad_updates, f_test,
            save=f_save, save_images=f_analysis,
            monitor=monitor,
            out_path=self.out_path,
            name=module.name,
            test_every=test_every,
            show_every=show_every,
            **learning_args)

    def run(self, module, model_to_load=None, **kwargs):
        self.setup(module)
        data = self.set_data(module)
        inputs = self.make_inputs(module, data)
        models, tparams = self.form_models(
            module, data, model_to_load=model_to_load)
        inputs, results, updates, constants, model_outs = self.set_cost(
            module, models, inputs)
        cost = results['cost']

        f_test = self.set_test_function(
            module, inputs, results, model_outs, updates)
        f_save = self.set_save_function(module, tparams)
        f_analysis = self.set_analysis_function(
            module, inputs, data, results, models, model_outs, updates)

        kwargs.update(
            f_test=f_test,
            f_save=f_save,
            f_analysis=f_analysis
        )

        self.train(module, inputs, cost, tparams, updates, constants, data, **kwargs)


class ModuleContainer(object):
    __required_methods = ['create_model', 'set_cost']
    __optional_methods = ['setup', 'set_data', 'set_test_function',
                          'set_save_function', 'set_analysis_function']
    __required_arguments = ['learning', 'dataset']

    def __init__(self, module_path, name=None):
        if name is None:
            self.name = '.'.join(module_path.split('/')[-1].split('.')[:-1])
        else:
            self.name = name
        module = imp.load_source(self.name, module_path)

        try:
            self.resolve_dataset = module.resolve_dataset
        except AttributeError:
            self._raise_no_attribute('resolve_dataset')

        self.set_arguments(module)

        for method in self.__required_methods + self.__optional_methods:
            try:
                self.set_method(module, method)
            except AttributeError:
                if method in self.__required_arguments:
                    self._raise_no_attribute(method)

        self.tparams = None
        self.models = None

    def _raise_no_attribute(self, method):
            raise AttributeError('No required `%s` method or import '
                'found in module %s.' % (method, self.name))

    def set_method(self, module, method):
        logger.debug('Setting method `%s` from module' % method)
        setattr(self, method, getattr(module, method))

    def set_arguments(self, module):
        logger.debug('Settting arguments')
        try:
            module_arg_keys = module.arg_keys
        except AttributeError:
            module_arg_keys = []

        self.args = dict(extra=dict())
        arg_keys = self.__required_arguments[:]
        arg_keys = list(set(arg_keys + module_arg_keys))
        for arg_key in arg_keys:
            default = 'default_' + arg_key + '_args'
            if not hasattr(module, default):
                raise ImportError('Module %s must define %s'
                                  % (self.name, default))
            self.args[arg_key] = getattr(module, default)
        logger.debug('Module default arguments are %s'
                     % pprint.pformat(self.args))

    def update(self, exp_dict):
        for key in exp_dict.keys():
            if key.endswith('_args'):
                k = key[:-5]
            else:
                continue
            if k in self.args.keys():
                v = exp_dict.pop(key)
                extra_keys = list(set(v.keys()) - set(self.args[k].keys()))
                if len(extra_keys) > 0:
                    raise KeyError('Extra keys found: %s' % extra_keys)
                self.args[k].update(**v)
                logger.debug('Updating %s arguments with %s'
                             % (k, pprint.pformat(v)))


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = make_argument_parser_trainer()
    args = parser.parse_args()
    exp_dict = set_experiment(args)
    module = ModuleContainer(
        path.abspath(exp_dict.pop('module')),
        name=exp_dict.pop('name', None))
    module.update(exp_dict)

    show_every = exp_dict.pop('show_every', None)
    test_every = exp_dict.pop('test_every', None)
    model_to_load = exp_dict.pop('model_to_load', None)

    trainer = Trainer(module, **exp_dict)
    trainer.run(module, show_every=show_every, test_every=test_every,
                model_to_load=model_to_load)