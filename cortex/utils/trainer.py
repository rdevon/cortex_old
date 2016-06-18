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

from . import floatX, intX
from ..models import Layer
from .monitor import SimpleMonitor
from .preprocessor import Preprocessor
from .tools import _p, load_model as _load_model, print_profile, print_section
from .training import (
    main_loop, make_argument_parser_trainer, set_experiment, set_model,
    set_optimizer, set_params)
from ..datasets import dataset_factory


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

def make_inputs(module):
    '''Forms the inputs from the dataset

    '''
    dataset = module.dataset

    print_section('Setting inputs')
    d = dataset.next()
    inps = OrderedDict()
    for k, v in d.iteritems():
        logger.debug('Data mode %s has batch shape %s.' % (k, v.shape))
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

        X = C(k, dtype=dtype)
        inps[k] = X
    logger.debug('Dataset has the following inputs: %s with types %s'
                 % (inps, [inp.dtype for inp in inps.values()]))
    dataset.reset()
    module.inputs = inps

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
          f_viz=None, f_outs=None, test_every=10, show_every=10,
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
        save=f_save, save_images=f_viz, f_outs=f_outs,
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

        check(module)
        finish(module)

        kwargs.update(
            f_test=f_test,
            f_save=f_save,
            f_viz=f_viz,
            f_outs=f_outs
        )

        train(module, cost, tparams, updates, constants, **kwargs)


class Inspector(object):
    def __init__(self, module, model_to_load=None):
        if isinstance(module, str):
            module = load_module(module)
        self.module = module
        self.set()

    def set(self, **kwargs):
        eval_methods = set_eval_functions(self.module, **kwargs)
        self.eval_keys = []
        for k, v in eval_methods.iteritems():
            self.eval_keys.append(k)
            setattr(self, k, v)

    def show(self):
        for k in self.eval_keys:
            self.__dict__[k]()

class ModuleContainer(object):
    __required_methods = ['_build', '_cost']
    __optional_methods = ['_setup', '_data', '_test', '_save', '_viz', '_check',
                          '_eval', '_finish', '_analysis']
    __required_arguments = ['_learning_args', '_dataset_args']
    _save_fields = ['name', 'preprocessing', 'module_path']

    def __init__(self, module_path, out_path, preprocessing=None, name=None):
        self.logger = logger
        self.preprocessing = preprocessing
        self.preprocessor = Preprocessor(self.preprocessing)

        if name is None:
            name = '.'.join(module_path.split('/')[-1].split('.')[:-1])
        self.name = name
        self.out_path = out_path
        self.module_path = module_path
        module = imp.load_source(self.name, self.module_path)

        for arg in self.__required_arguments + self.__required_methods:
            if not hasattr(module, arg):
                self._raise_no_attribute(arg)
        try:
            self.resolve_dataset = module.resolve_dataset
        except AttributeError:
            self._raise_no_attribute('resolve_dataset')

        self.models = None
        self.args = dict(extra=dict())
        self.dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.inputs = None
        self.set_methods(module)
        self.set_args(module)

    def _raise_no_attribute(self, method):
        required = self.__required_arguments + self.__required_methods
        raise AttributeError('No required `{method}` method or import '
            'found in module {module}. Please check that module {module} '
            'has {required}'.format(method=method, module=self.name,
                                    required=required))

    def check_protected(self, key):
        if key in self.__dict__.keys():
            raise KeyError('Module already has member or method {key}. If {key}'
                           'is a parameter name, please rename it.'.format(
                            key=key))

    def set_methods(self, module):
        for method in self.__required_methods + self.__optional_methods:
            if hasattr(module, method):
                m = method[1:]
                logger.debug('Setting method `%s` from module' % m)
                setattr(self, m, types.MethodType(getattr(module, method), self))

    def set_args(self, module):
        self.args.clear()
        logger.info('Settting arguments')
        try:
            module_arg_keys = module.extra_arg_keys
        except AttributeError:
            module_arg_keys = []

        arg_keys = self.__required_arguments[:] + ['_model_args']
        arg_keys = list(set(arg_keys + module_arg_keys))
        for arg_key in arg_keys:
            self.check_protected(arg_key)
            if not hasattr(module, arg_key) and arg_key != '_model_args':
                raise ImportError('Module %s must define %s'
                                  % (self.name, arg_key))
            args = getattr(module, arg_key)
            if arg_key in ['_learning_args', '_dataset_args', '_model_args']:
                arg_key = arg_key[1:]
            self.args[arg_key] = args
            if arg_key == 'model_args':
                for k, v in args.iteritems():
                    self.__dict__[k] = v
            else:
                self.__dict__[arg_key] = self.args[arg_key]
        logger.debug('Module default arguments are %s'
                     % pprint.pformat(self.args))
        self.learning_args = self.args['learning_args']

    def update(self, exp_dict):
        for key in exp_dict.keys():
            if not key.endswith('_args'):
                continue
            if key in self.args.keys():
                v = exp_dict.pop(key)
                self.args[key].update(**v)
                logger.info('Updating %s arguments with %s'
                             % (key, pprint.pformat(v)))
                if key == 'model_args':
                    self.__dict__.update(**v)


def flatten_component_layers(models, model_dict):
    component_list = []
    def add_component(component):
        if component is None:
            return
        if not isinstance(component, Layer):
            raise TypeError('Components must be a subtype of `Layer` or list '
                            'of `Layer` (%s)' % component)
        if component.name in model_dict.keys():
            raise ValueError('Duplicate key found: %s' % component.name)
        model_dict[component.name] = component
        component_list.append(component)

    for model in models:
        if hasattr(model, '_components'):
            components = [model.__dict__[c] for c in model._components]
            for component in components:
                if isinstance(component, list):
                    for c in component:
                        add_component(c)
                else:
                    add_component(component)
    if len(component_list) > 0:
        flatten_component_layers(component_list, model_dict)

def load_module(model_file, strict=True):
    '''Loads pretrained model.

    Args:
        model_file (str): path to file.
        strict (bool): fail on extra parameters.

    Returns:
        ModuleContainer: module container.
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

    try:
        module_path = d.pop('module_path')
        name = d.pop('name')
        preprocessing = d.pop('preprocessing')
    except KeyError:
        raise TypeError('Model file does not contain the appropriate fields '
                        'to be loaded as a module.')
    out_path = '/'.join(module_path.split('/')[:-1])

    module = ModuleContainer(module_path, out_path, name=name,
                             preprocessing=preprocessing)

    pretrained_kwargs = dict()
    arg_kwargs = dict()
    for k, v in d.iteritems():
        if k.endswith('_args'):
            arg_kwargs[k] = v
        else:
            pretrained_kwargs[k] = v
    module.update(arg_kwargs)

    setup(module)
    set_data(module)
    make_inputs(module)
    build(module)

    logger.info('Pretrained model(s) has the following parameters: \n%s'
          % pprint.pformat(pretrained_kwargs.keys()))

    flatten_component_layers(module.models.values(), module.models)

    for model in module.models.values():
        if model is None:
            continue
        logger.info('---Loading params for %s' % model.name)
        for k, v in model.params.iteritems():
            try:
                param_key = _p(model.name, k)
                pretrained_v = pretrained_kwargs.pop(param_key)
                logger.info('Found %s for %s %s'
                            % (k, model.name, pretrained_v.shape))
                assert model.params[k].shape == pretrained_v.shape, (
                    'Sizes do not match: %s vs %s'
                    % (model.params[k].shape, pretrained_v.shape)
                )
                model.params[k] = pretrained_v
            except KeyError:
                try:
                    param_key = '{key}'.format(key=k)
                    pretrained_v = pretrained_kwargs[param_key]
                    logger.info('Found %s, but name is ambiguous' % k)
                    assert model.params[k].shape == pretrained_v.shape, (
                        'Sizes do not match: %s vs %s'
                        % (model.params[k].shape, pretrained_v.shape)
                    )
                    model.params[k] = pretrained_v
                except KeyError:
                    logger.info('{} not found'.format(k))

    if len(pretrained_kwargs) > 0 and strict:
        raise ValueError('ERROR: Leftover params: %s' %
                         pprint.pformat(pretrained_kwargs.keys()))
    elif len(pretrained_kwargs) > 0:
        logger.warn('Leftover params: %s' %
                      pprint.pformat(pretrained_kwargs.keys()))

    set_tparams(module)
    return module

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