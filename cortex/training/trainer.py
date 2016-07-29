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
from ..utils.logger import get_class_logger
from ..utils.tools import check_bad_nums, update_dict_of_lists


'''
except KeyboardInterrupt:
    print 'Training interrupted.'
except:
    logger.exception('Exception reached during training')
    raise

try:
    if out_path is not None:
        outfile = path.join(out_path, '{name}_{t}.npz'.format(name=name, t=int(time.time())))
        last_outfile = path.join(out_path, '{name}_last.npz'.format(name=name))

        if save is not None:
            logging.info('Saving')
            save(outfile)
            save(last_outfile)
            logging.info('Done saving.')
except KeyboardInterrupt:
    print 'Saving interupted.'

print 'Main loop finished.'
'''


class Trainer(object):
    '''Trainer class for cortex.

    Trainer processes and organizes scripts for use in cortex.

    '''

    def __init__(self, session, name='trainer', data_mode='train',
                 optimizer=None, epochs=None, batch_size=None,
                 learning_rate=None, optimizer_args=None):
        if optimizer is None:
            raise TypeError('`optimizer` not set')
        if epochs is None:
            raise TypeError('`epochs` not set')
        if batch_size is None:
            raise TypeError('`batch_size` not set')
        if learning_rate is None:
            raise TypeError('`learning_rate` not set')

        epoch_t0 = time.time()
        training_time = 0
        self.logger = get_class_logger(self)
        self.session = session
        self.set_optimizer(optimizer=optimizer, optimizer_args=optimizer_args)
        self.learning_rate = learning_rate
        self.epoch = 0
        self.epochs = epochs
        self.batch_size = batch_size
        self.name = name
        self.data_mode = data_mode
        self.training_time = 0

        '''
        if f_extra is not None:
            logging.info('Performing initial evaluation function...')
            f_extra()
        '''

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
                    return
                else:
                    self.start_pbar(n)
                    self.session.reset_data(mode=self.data_mode)
            self.epoch_pbar.update(self.session.data_pos)
            rval = self.f_grad_shared(*inputs)

            if check_bad_nums(rval):
                check_bad_nums(f_test(*inputs))
                if f_outs is not None:
                    check_bad_nums(f_outs(*inps))
                raise RuntimeError('Dying, found bad cost... Sorry (bleh)')

            self.f_grad_updates(self.learning_rate)


    def set_optimizer(self, optimizer='sgd', optimizer_args=None):
        '''Sets the parameter update functions with optimizer.

        Args:
            optimizer (Optional[str]): optimizer string. See `utils.op` for details.
                Defaults to `sgd`.
            optimizer_args (Optional[dict]): optional arguments for optimizer.

        '''
        self.logger.info('Setting up optimizer `%s`' % optimizer)

        session = self.session

        from ..manager import get_manager
        manager = get_manager()

        cost = session.cost
        tparams = manager.tparams

        if optimizer_args is None:
            optimizer_args = dict()

        grads = T.grad(
            cost, wrt=tparams.values(), consider_constant=session.constants)
        grads = OrderedDict((k, g) for k, g in zip(tparams.keys(), grads))

        if optimizer not in _ops.keys():
            raise KeyError('Optimizer `%s` not found, available: %s'
                           % (optimizer, _ops.keys()))
        optimizer = _ops[optimizer]

        lr = T.scalar(name='lr')
        f_grad_shared, f_grad_updates = optimizer(
            lr, tparams, grads, session.inputs, cost, extra_ups=session.updates,
            **optimizer_args)

        self.f_grad_shared = f_grad_shared
        self.f_grad_updates = f_grad_updates