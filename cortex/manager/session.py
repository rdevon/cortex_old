'''Module for the Session class.

'''
from collections import OrderedDict
import logging
import pprint
import theano
from theano import tensor as T

from . import get_manager, is_tensor_arg, resolve_tensor_arg
from ..models import get_noise_switch
from ..utils import floatX
from ..utils.tools import _p


class Session(object):
    _idx = 0
    sessions = []
    noise_switch = get_noise_switch()

    def __init__(self, manager=None, noise=True, batch_size=None):
        if manager is None: manager = get_manager()
        self.logger = logging.getLogger(
            '.'.join([self.__module__, self.__class__.__name__]))
        self.idx = self._idx
        self._idx += 1
        self.sessions.append(self)
        self.manager = manager
        self.reset()
        self.noise = noise
        self.batch_size = batch_size

    @staticmethod
    def _reset():
        Session.sessions = []
        Session._idx = 0

    def reset(self):
        self.tensors = {}
        self.cost = T.constant(0.).astype(floatX)
        self.costs = {}
        self.stats = {}

        self.updates = theano.OrderedUpdates()
        self.constants = []
        self.inputs = []
        self.datasets = []
        self.input_keys = []
        self.data_pos = 0

    def add_tensors(self, out, key_prefix=None, what=None):
        for k, v in out.iteritems():
            if k == 'updates':
                self.updates += v

            elif k == 'constants':
                self.constants += v

            else:
                if what == 'cost':
                    if k == 'cost':
                        key = key_prefix
                        self.cost += v
                    else:
                        key = _p(key_prefix, k)
                    self.logger.debug('Adding cost `%s`' % key)
                    self.costs[key] = v
                elif what == 'stat':
                    if k == 'stat':
                        key = key_prefix
                    else:
                        key = _p(key_prefix, k)
                    self.stats[key] = v
                    self.logger.debug('Adding stat `%s`' % key)
                else:
                    key = _p(key_prefix, k)

                if key in self.tensors.keys():
                    raise KeyError('Cannot overwrite %s' % key)
                else:
                    self.logger.debug('Adding tensor `%s`' % key)
                    self.tensors[key] = v

        if _p(key_prefix, 'outputs') not in self.tensors.keys():
            self.tensors[_p(key_prefix, 'outputs')] = dict()

        self.tensors[_p(key_prefix, 'outputs')].update(**out)

    def add_cost(self, *args, **kwargs):
        self.add('cost', *args, **kwargs)

    def add_stat(self, *args, **kwargs):
        self.add('stat', *args, **kwargs)

    def add_step(self, *args, **kwargs):
        self.add('step', *args, **kwargs)

    def add(self, what, name=None, op=None, cell_name=None, constants=None,
            args=None, kwargs=None, test=False):
        self.logger.debug('Adding %s: %s' % (what,
            dict(op=op, name=name, cell_name=cell_name, constants=constants,
                 args=args, kwargs=kwargs)))

        args, kwargs = self.resolve_op_args(args, kwargs, constants=constants)
        if cell_name is not None:
            cell = self.manager.cells[cell_name]
            out = op(cell, *args, **kwargs)
        else:
            cell = None
            out = op(*args, **kwargs)

        if isinstance(out, T.TensorVariable):
            new_out = dict()
            if what == 'cost':
                new_out['cost'] = out
            elif what == 'stat':
                new_out['stat'] = out
            else:
                new_out['output'] = out

            out = new_out

        self.add_tensors(out, key_prefix=name, what=what)

        if test:
            self.reset_data()
            if cell is None or (cell is not None and cell._test_order is None):
                test_order = out.keys()
            else:
                test_order = cell._test_order

            batch_size = self.batch_size or 10
            data = self.next_batch(batch_size=batch_size)
            if what in ['cost', 'stat']:
                for k, o in out.iteritems():
                    self.logger.info('Testing stat with batchsize %d' % batch_size)
                    f = theano.function(self.inputs, o, updates=self.updates,
                                        on_unused_input='ignore')
                    self.test(data, f, key=k, key_prefix=name, cell=cell)
            else:
                for key in test_order:
                    if key in ['updates', 'constants']:
                        continue
                    self.logger.info('Testing `%s` from step %s with batchsize %d'
                                     % (key, name, batch_size))
                    t = out[key]
                    f = theano.function(self.inputs, t, updates=self.updates,
                                        on_unused_input='ignore')
                    self.test(data, f, key, name, cell=cell)

    def add_samples(self, name=None, op=None, dist_key=None, shape=None,
                    cell_name=None, kwargs=None):
        self.logger.debug('Adding samples: %s' %
            dict(op=op, name=name, dist_key=dist_key, shape=shape,
                 cell_name=cell_name, kwargs=kwargs))

        _, kwargs = self.resolve_op_args([], kwargs)

        if dist_key is not None:
            P = self.tensors[dist_key]
        else:
            P = None

        cell = self.manager.cells[cell_name]
        epsilon = cell.generate_random_variables(shape, P=P)
        self.tensors[name + '_epsilon'] = epsilon
        samples = cell._sample(epsilon, P=P, **kwargs)

        if isinstance (samples, T.TensorVariable):
            samples = {name: samples}
        else:
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

    def resolve_op_args(self, args, kwargs, constants=None):
        manager = self.manager
        tensors = self.tensors

        if args is None: args = []
        if kwargs is None: kwargs = {}
        if constants is None: constants = []
        new_args = []
        for arg in args:
            if arg in self.tensors.keys():
                ten = tensors[arg]
                if arg in constants:
                    ten = ten.copy()
                    tensors[arg + '(copy)'] = ten
                    self.constants.append(ten)
                new_args.append(ten)
            elif is_tensor_arg(arg):
                name, key, _ = resolve_tensor_arg(arg)
                if name in manager.datasets.keys():
                    dataset_tensor = manager.datasets[name]['tensors'][key]
                    tensors[arg] = dataset_tensor
                    self.inputs.append(dataset_tensor)
                    self.datasets.append(name)
                    self.input_keys.append(key)

                if arg not in tensors.keys() and arg in manager.samples.keys():
                    self.add_samples(**manager.samples[arg])

                if arg in tensors.keys():
                    ten = tensors[arg]
                    if arg in constants:
                        ten = ten.copy()
                        tensors[arg + '(copy)'] = ten
                        self.constants.append(ten)
                    new_args.append(ten)
                elif arg not in manager.cells.keys():
                    for tpk, tparam in manager.tparams.iteritems():
                        if arg in tpk:
                            new_args.append(tparam)
                else:
                    raise ValueError('Could not find tensor %s, found: %s'
                                     % (arg, tensors.keys()))
            else:
                new_args.append(arg)
        assert len(new_args) >= len(args), (args, new_args)

        new_kwargs = {}
        for key, arg in kwargs.iteritems():
            if arg in self.tensors.keys():
                arg = self.tensors[arg]
            elif is_tensor_arg(arg):
                name, key_, _ = resolve_tensor_arg(arg)
                if name in manager.datasets.keys():
                    if arg not in tensors.keys():
                        dataset_tensor = manager.datasets[name]['tensors'][key_]
                        tensors[arg] = dataset_tensor
                        self.inputs.append(dataset_tensor)
                        self.datasets.append(name)
                        self.input_keys.append(key_)

                if arg not in tensors.keys() and arg in manager.samples.keys():
                    self.add_samples(**manager.samples[arg])

                if arg in tensors.keys():
                    arg = tensors[arg]
                else:
                    raise ValueError('Could not find tensor %s' % arg)

            if key == 'inputs' and isinstance(arg, dict):
                new_kwargs.update(**arg)
            else:
                new_kwargs[key] = arg

        return new_args, new_kwargs

    def build(self, test=False):
        manager = self.manager
        tensors = self.tensors
        manager._current_session = self

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
                self.add_samples(**samples)

        manager._current_session = None

    def get_dataset_names(self):
        seen = set()
        dataset_names = [x for x in self.datasets
                         if not (x in seen or seen.add(x))]
        return dataset_names

    def get_dataset_size(self, mode=None):
        dataset_names = self.get_dataset_names()
        n = None
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
            if n is None:
                n = dataset[m].n_samples
            else:
                n = min(n, dataset[m].n_samples)

        return n

    def next_batch(self, mode=None, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        if batch_size is None:
            raise TypeError('`batch_size` keyword must be set.')

        data = []
        batches = {}
        dataset_names = self.get_dataset_names()

        try:
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

                batch = dataset[m].next(batch_size)
                batches[name] = batch
                if name == dataset_names[0]:
                    self.data_pos = dataset[m].pos
                else:
                    if self.data_pos != -1 and self.data_pos != dataset[m].pos:
                        raise ValueError('Dataset position mismatch. (%d vs %d)'
                                         % (self.data_pos, dataset[m].pos))
        except StopIteration:
            for name in dataset_names:
                if mode is None:
                    ms = dataset.keys()
                    if 'train' in ms:
                        m = 'train'
                    else:
                        ms.pop('dims', 'dimensions', 'idx')
                        m = ms[0]
                else:
                    m = mode
                dataset[m].reset()
            raise StopIteration

        for name, key in zip(self.datasets, self.input_keys):
            batch = batches[name]
            data.append(batch[key])

        if self.data_pos == -1:
            self.data_pos = self.get_dataset_size(mode=mode)

        return data

    def reset_data(self, mode=None):
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

            dataset[m].reset()
        self.data_pos = 0