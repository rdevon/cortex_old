'''Module for the Session class.

'''
from collections import OrderedDict
import logging
import pprint
import theano
from theano import tensor as T

from . import get_manager, is_tensor_arg, resolve_tensor_arg
from ..models import get_noise_switch


class Session(object):
    _idx = 0
    sessions = []
    noise_switch = get_noise_switch()

    def __init__(self, manager=None, noise=True):
        if manager is None: manager = get_manager()
        self.logger = logging.getLogger(
            '.'.join([self.__module__, self.__class__.__name__]))
        self.idx = self._idx
        self._idx += 1
        self.sessions.append(self)
        self.manager = manager
        self.reset()
        self.noise = noise

    @staticmethod
    def _reset():
        Session.sessions = []
        Session._idx = 0

    def reset(self):
        self.tensors = {}
        self.costs = []
        self.stats = {}

        self.updates = theano.OrderedUpdates()
        self.constants = []
        self.inputs = []
        self.datasets = []
        self.input_keys = []

    def add_tensors(self, out, key_prefix=None):

        for k, v in out.iteritems():
            if k == 'updates':
                self.updates += v
            elif k == 'constants':
                self.constants += v
            else:
                if key_prefix is None:
                    key = k
                else:
                    key = key_prefix + '.' + k
                if key in self.tensors.keys():
                    raise KeyError('Cannot overwrite %s' % key)
                else:
                    self.tensors[key] = v

    def add_step(self, op=None, cell_name=None, args=None, kwargs=None,
                 test=False):
        self.logger.debug('Adding step: %s' % pprint.pformat(
            dict(op=op, cell_name=cell_name, args=args, kwargs=kwargs)))

        args, kwargs = self.resolve_op_args(args, kwargs)

        if cell_name is not None:
            cell = self.manager.cells[cell_name]
            out = op(cell, *args, **kwargs)
            key_prefix = cell_name
        else:
            name = op.__name__
            out = op(*args, **kwargs)
            key_prefix = name
            cell = None

        if isinstance(out, T.TensorVariable):
            out = dict(output=out)

        self.add_tensors(out, key_prefix=key_prefix)

        if test:
            if cell is None or (cell is not None and cell._test_order is None):
                test_order = out.keys()
            else:
                test_order = cell._test_order

            data = self.next()

            for key in test_order:
                self.logger.info('Testing `%s` from step %s' % (key, key_prefix))
                t = out[key]
                f = theano.function(self.inputs, t, updates=self.updates)
                self.test(data, f, key, key_prefix, cell=cell)

    def add_cost(self, name=None, op=None, cell_name=None, args=None,
                 kwargs=None, test=False):
        self.logger.debug('Adding cost: %s' % pprint.pformat(
            dict(name=name, op=op, cell_name=cell_name, args=args,
                 kwargs=kwargs)))

        args, kwargs = self.resolve_op_args(args, kwargs)
        if cell_name is not None:
            cell = self.manager.cells[cell_name]
            out = op(cell, *args, **kwargs)
        else:
            cell = None
            out = op(*args, **kwargs)

        self.costs.append(out)
        self.stats[name] = out

        if test:
            data = self.next()
            self.logger.info('Testing cost')
            f = theano.function(self.inputs, out, updates=self.updates)
            self.test(data, f, key=name, key_prefix='cost', cell=cell)

    def add_stat(self, name=None, op=None, cell_name=None, args=None,
                 kwargs=None, test=False):
        self.logger.debug('Adding stat: %s' % pprint.pformat(
            dict(name=name, op=op, cell_name=cell_name, args=args,
                 kwargs=kwargs)))

        args, kwargs = self.resolve_op_args(args, kwargs)
        if cell_name is not None:
            cell = self.manager.cells[cell_name]
            out = op(cell, *args, **kwargs)
        else:
            cell = None
            out = op(*args, **kwargs)

        self.stats[name] = out

        if test:
            data = self.next()
            self.logger.info('Testing stat')
            f = theano.function(self.inputs, out, updates=self.updates)
            self.test(data, f, key=name, key_prefix='stat', cell=cell)

    def add_samples(self, name=None, op=None, key=None, shape=None,
                    cell_name=None, kwargs=None):
        _, kwargs = self.resolve_op_args([], kwargs)

        if key is not None:
            P = self.tensors[key]
        else:
            P = None

        cell = self.manager.cells[cell_name]
        epsilon = cell.generate_random_variables(shape, P=P)
        self.tensors[name + '_epsilon'] = epsilon
        samples = cell._sample(epsilon, P=P, **kwargs)

        if isinstance (samples, T.TensorVariable):
            samples = dict(samples=samples)

        self.logger.debug('Adding samples: %s'
                          % pprint.pformat(dict(samples)))

        samples = dict(
            (k + '(sampling)', v)
             if not k in ['samples', 'updates', 'constants']
             else (k, v)
            for k, v in samples.iteritems()
        )
        self.add_tensors(samples, key_prefix=cell_name)

    def test(self, data, f, key, key_prefix, cell=None):
        try:
            t_ = f(*data)
            self.logger.info('Tensor `%s` for `%s` has shape %s '
                             '(passes without error)'
                             % (key, key_prefix, t_.shape))
        except ValueError as e:
            self.logger.error(
                'Test function failed for tensor `%s` in `%s`'
                % (key, key_prefix))
            if cell is not None:
                self.logger.info('Cell: %s' % cell)

            for d in data:
                self.logger.info('Data shape: %s' % (d.shape,))

            raise e

    def resolve_op_args(self, args, kwargs):
        manager = self.manager
        tensors = self.tensors
        if args is None: args = []
        if kwargs is None: kwargs = {}
        new_args = []
        for arg in args:
            if is_tensor_arg(arg):
                name, key, _ = resolve_tensor_arg(arg)
                if name in manager.datasets.keys():
                    if arg not in tensors.keys():
                        dataset_tensor = manager.datasets[name]['tensors'][key]
                        tensors[arg] = dataset_tensor
                        self.inputs.append(dataset_tensor)
                        self.datasets.append(name)
                        self.input_keys.append(key)

                if arg not in tensors.keys() and arg in manager.samples.keys():
                    self.add_samples(name=arg, **manager.samples[arg])

                if arg in tensors.keys():
                    new_args.append(tensors[arg])
                else:
                    raise ValueError('Could not find tensor %s, found: %s'
                                     % (arg, tensors.keys()))
            else:
                new_args.append(arg)

        new_kwargs = {}
        for key, arg in kwargs.iteritems():
            if is_tensor_arg(arg):
                name, key_, _ = resolve_tensor_arg(arg)
                if name in manager.datasets.keys():
                    if arg not in tensors.keys():
                        dataset_tensor = manager.datasets[name]['tensors'][key_]
                        tensors[arg] = dataset_tensor
                        self.inputs.append(dataset_tensor)
                        self.datasets.append(name)
                        self.input_keys.append(key_)

                if arg in tensors.keys():
                    new_kwargs[key] = tensors[arg]
                else:
                    raise ValueError('Could not find tensor %s' % arg)
            else:
                new_kwargs[key] = arg
        return new_args, new_kwargs

    def build(self, test=False):
        manager = self.manager
        tensors = self.tensors

        if self.noise:
            self.noise_switch.switch('on')
        else:
            self.noise_switch.switch('off')

        for step in manager.steps:
            self.add_step(test=test, **step)

        for name, cost in manager.costs.iteritems():
            self.add_cost(name=name, test=test, **cost)

        for name, stat in manager.stats.iteritems():
            self.add_stat(name=name, test=test, **stat)

        for name, samples in manager.samples.iteritems():
            if name not in tensors.keys():
                self.add_samples(name=name, **samples)

    def get_dataset_names(self):
        seen = set()
        dataset_names = [x for x in self.datasets
                         if not (x in seen or seen.add(x))]
        return dataset_name

    def get_dataset_batches(self, mode=None):
        dataset_names = self.get_dataset_names()
        n = float('infty')
        for name in dataset_names:


    def next(self, mode=None, batch_size=None):
        data = []
        batches = {}
        dataset_names = self.get_dataset_names()

        for name in dataset_names:
            dataset = self.manager.datasets[name]
            if mode is None:
                ms = dataset.keys()
                if 'train' in ms:
                    m = 'train'
                else:
                    ms.pop('dims', 'dimensions', 'idx')
                    m = ms[0]
            else:
                m = mode

            batch = dataset[m].next(batch_size=batch_size)
            batches[name] = batch

        for name, key in zip(self.datasets, self.input_keys):
            batch = batches[name]
            data.append(batch[key])

        return data

    def reset_data(self, mode=None):
        dataset_names = self.get_dataset_names()

        if mode is None:
            ms = dataset.keys()
            if 'train' in ms:
                m = 'train'
            else:
                ms.pop('dims', 'dimensions', 'idx')
                m = ms[0]

        self.datasets[m].reset()