'''Module for the Session class.

'''
from collections import OrderedDict
import theano
from theano import tensor as T

from . import get_manager, is_tensor_arg, resolve_tensor_arg


class Session(object):
    _idx = 0
    sessions = []

    def __init__(self, manager=None):
        if manager is None: manager = get_manager()
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
                name, key, _ = resolve_tensor_args(arg)
                if name in manager.datasets.keys():
                    tensors[arg] = manager.datasets[name]['tensors'][key]

                if arg in tensors.keys():
                    new_kwargs[key] = tensors[arg]
                else:
                    raise ValueError('Could not find tensor %s' % arg)
            else:
                new_kwargs[key] = arg

        return new_args, new_kwargs

    def build(self):
        manager = self.manager
        tensors = self.tensors

        for step in manager.steps:
            op = step['op']
            cell_name = step['cell_name']
            if cell_name is not None:
                cell = manager.cells[cell_name]
            else:
                cell = None
                name = op.__name__

            args, kwargs = self.resolve_op_args(step['args'], step['kwargs'])

            if cell is None:
                out = op(*args, **kwargs)
            else:
                out = op(cell, *args, **kwargs)
            if isinstance(out, T.TensorVariable):
                out = dict(output)

            if cell is not None:
                key_prefix = cell_name
            else:
                key_prefix = name

            for k, v in out.iteritems():
                key = key_prefix + '.' + k
                if key in self.tensors.keys():
                    raise KeyError('Cannot overwrite %s' % key)
                else:
                    self.tensors[key] = v

        for cost in manager.costs:
            op = cost['op']
            args, kwargs = self.resolve_op_args(cost['args'], cost['kwargs'])
            out = op(*args, **kwargs)
            self.costs.append(out)

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