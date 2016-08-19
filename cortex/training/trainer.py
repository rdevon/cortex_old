'''Trainer class.

Does typical training work.

'''

from collections import OrderedDict
import logging
import numpy as np
from progressbar import (
    Bar,
    ProgressBar,
    Percentage,
    Timer
)
import theano
from theano import tensor as T
import time

from .op import _ops
from ..utils import floatX
from ..utils.logger import get_class_logger
from ..utils.tools import check_bad_nums, update_dict_of_lists


class Trainer(object):
    '''Trainer class for cortex.

    Trainer processes and organizes scripts for use in cortex.

    '''

    def __init__(self, session, name='trainer', data_mode='train',
                 optimizer=None, epochs=None, batch_size=None,
                 learning_rate=None, optimizer_args=None, costs=None,
                 models=None, excludes=None):
        if optimizer is None:
            raise TypeError('`optimizer` not set')
        if epochs is None:
            raise TypeError('`epochs` not set')
        if batch_size is None and session.batch_size is None:
            raise TypeError('`batch_size` not set')
        if learning_rate is None:
            raise TypeError('`learning_rate` not set')

        epoch_t0 = time.time()
        training_time = 0
        self.logger = get_class_logger(self)
        self.session = session
        self.learning_rate = learning_rate
        self.epoch = 0
        self.epochs = epochs
        self.batch_size = batch_size
        self.name = name
        self.data_mode = data_mode
        self.training_time = 0
        self.u = 0

        self.f_grads = []
        self.f_updates = []
        self.f_freqs = []
        self.tparams = []

        self.optimizer = optimizer
        self.optimizer_args = optimizer_args
        self.excludes = excludes or []

    def start_pbar(self, n):
        widgets = ['Epoch {epoch} (training {name}, '.format(
            epoch=self.epoch, name=self.name), Timer(), ' Total time (%.2f)): '
                   % self.training_time, Bar()]
        self.epoch_pbar = ProgressBar(widgets=widgets, maxval=n).start()

    def next_epoch(self, n_epochs=1):
        t0 = time.time()

        self.session.reset_data(mode=self.data_mode)
        n = self.session.get_dataset_size(mode=self.data_mode)
        self.start_pbar(n)

        start_epoch = self.epoch
        grads = OrderedDict()

        while True:
            try:
                inputs = self.session.next_batch(
                    mode=self.data_mode, batch_size=self.batch_size)
            except StopIteration:
                t1 = time.time()
                self.training_time += t1 - t0
                t0 = time.time()

                self.epoch += 1
                if self.epoch >= self.epochs:
                    print
                    raise StopIteration
                elif self.epoch >= start_epoch + n_epochs:
                    print
                    return grads
                else:
                    self.start_pbar(n)
                    self.session.reset_data(mode=self.data_mode)
            self.epoch_pbar.update(self.session.data_pos)

            for f_grad, f_update, freq in zip(
                self.f_grads, self.f_updates, self.f_freqs):
                if self.u % freq != 0: continue

                rval = f_grad(*inputs)
                _grads = dict((k, v) for k, v in rval.iteritems()
                    if k.startswith('_grad_'))
                grads.update(**_grads)

                if check_bad_nums(rval):
                    raise RuntimeError('Dying, found bad cost... Sorry (bleh)')

                f_update(self.learning_rate)

            self.u += 1

    def set_optimizer(self, cost=None, models=None, freq=None):
        '''Sets the parameter update functions with optimizer.

        Args:
            optimizer (Optional[str]): optimizer string. See `utils.op` for details.
                Defaults to `sgd`.
            optimizer_args (Optional[dict]): optional arguments for optimizer.

        '''
        from .. import _manager as manager
        optimizer = self.optimizer
        optimizer_args = self.optimizer_args or dict()

        if optimizer not in _ops.keys():
            raise KeyError('Optimizer `%s` not found, available: %s'
                           % (optimizer, _ops.keys()))

        session = self.session
        optimizer = _ops[optimizer]

        if cost is None:
            self.logger.debug('Using global cost for models %s' % models)
            cost = session.cost
        else:
            self.logger.debug('Using cost `%s` for models %s' % (cost, models))
            if isinstance(cost, list):
                cost_ = T.constant(0).astype(floatX)
                for cost__ in cost:
                    cost_ += session.costs[cost__]
                cost = cost_
            else:
                cost = session.costs[cost]

        if models is None:
            tparams = manager.tparams
        else:
            tparams = OrderedDict()
            if isinstance(models, str): models = [models]
            for model in models:
                for k, v in manager.tparams.iteritems():
                    prefix = '.'.join(k.split('.')[:-1])
                    if model == prefix: tparams[k] = v
        tparams = OrderedDict((k, v) for k, v in tparams.iteritems()
            if k not in self.excludes)

        self.logger.info('Computing gradients for params: %s' % tparams.keys())
        grads = T.grad(
            cost, wrt=tparams.values(), consider_constant=session.constants)
        grads = OrderedDict((k, g) for k, g in zip(tparams.keys(), grads))

        lr = T.scalar(name='lr')
        f_grad, f_update = optimizer(
            lr, tparams, grads, session.inputs, session.cost,
            extra_ups=session.updates, **optimizer_args)

        self.f_grads.append(f_grad)
        self.f_updates.append(f_update)
        self.f_freqs.append(freq or 1)
        self.tparams = list(set(self.tparams + tparams.keys()))