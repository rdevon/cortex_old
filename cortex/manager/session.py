'''Module for the Session class.

'''
from collections import OrderedDict
import logging
import pprint
import theano
from theano import tensor as T

from . import get_manager, is_tensor_arg, resolve_tensor_arg


class Session(object):
    _idx = 0
    sessions = []

    def __init__(self, manager=None):
        if manager is None: manager = get_manager()
        self.logger = logging.getLogger(
            '.'.join([self.__module__, self.__class__.__name__]))
        self.idx = self._idx
        self._idx += 1
        self.sessions.append(self)

        self.manager = manager
        self.reset()

    @staticmethod
    def _reset():
        Session.sessions = []
        Session._idx = 0

    def reset(self):
        self.tensors = {}
        self.costs = []
        self.updates = theano.OrderedUpdates()
        self.constants = []
        self.inputs = []
        self.datasets = []
        self.input_keys = []

    def resolve_op_args(self, args, kwargs):
        manager = self.manager
        tensors = self.tensors
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

                if arg in tensors.keys():
                    new_args.append(tensors[arg])
                else:
                    raise ValueError('Could not find tensor %s' % arg)
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

        for step in manager.steps:
            self.logger.debug('Adding step: %s' % pprint.pformat(dict(step)))
            op = step['op']
            args, kwargs = self.resolve_op_args(step['args'], step['kwargs'])

            cell_name = step['cell_name']
            if cell_name is not None:
                cell = manager.cells[cell_name]
                out = op(cell, *args, **kwargs)
                key_prefix = cell_name
            else:
                name = op.__name__
                out = op(*args, **kwargs)
                key_prefix = name
                cell = None

            if isinstance(out, T.TensorVariable):
                out = dict(output)

            for k, v in out.iteritems():
                if k == 'updates':
                    self.updates += v
                elif k == 'constants':
                    self.constants += v
                else:
                    key = key_prefix + '.' + k
                    if key in self.tensors.keys():
                        raise KeyError('Cannot overwrite %s' % key)
                    else:
                        self.tensors[key] = v

            if test:
                data = self.next()

                if cell is None or (cell is not None and cell._test_order is None):
                    test_order = out.keys()
                else:
                    test_order = cell._test_order

                for key in test_order:
                    self.logger.info('Testing `%s` from step %s' % (key, key_prefix))
                    t = out[key]
                    f = theano.function(self.inputs, t, updates=self.updates)
                    try:
                        t_ = f(*data)
                        self.logger.info('Tensor `%s` for `%s` has shape %s'
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

        for cost in manager.costs:
            self.logger.debug('Adding cost: %s' % pprint.pformat(dict(cost)))
            op = cost['op']
            args, kwargs = self.resolve_op_args(cost['args'], cost['kwargs'])
            cell_name = cost['cell_name']
            if cell_name is not None:
                cell = manager.cells[cell_name]
                out = op(cell, *args, **kwargs)
            else:
                out = op(*args, **kwargs)

            self.costs.append(out)

            if test:
                self.logger.info('Testing cost')
                data = self.next()

                f = theano.function(self.inputs, out, updates=self.updates)
                try:
                    t_ = f(*data)
                    self.logger.info('Tensor `%s` for `%s` has shape %s'
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

    def next(self, mode=None, batch_size=None):
        data = []
        batches = {}
        seen = set()
        dataset_names = [x for x in self.datasets
                         if not (x in seen or seen.add(x))]

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