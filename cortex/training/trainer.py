'''Trainer class.

Does typical training work.

'''

from collections import OrderedDict
import logging
import numpy as np
from progressbar import Bar, ProgressBar, Percentage, Timer
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
        if optimizer is None: raise TypeError('`optimizer` not set')
        if epochs is None: raise TypeError('`epochs` not set')
        if batch_size is None and session.batch_size is None:
            raise TypeError('`batch_size` not set')
        if learning_rate is None: raise TypeError('`learning_rate` not set')

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
                _grads = dict((k, v) for k, v in rval.items()
                    if k.startswith('_grad_'))
                grads.update(**_grads)

                if check_bad_nums(rval, self.logger):
                    raise RuntimeError('Dying, found bad cost... Sorry (bleh)')

                f_update(self.learning_rate)

            self.u += 1

    def set_optimizer(self, *model_costs, **kwargs):
        '''Sets the parameter update functions with optimizer.

        Args:
            optimizer (Optional[str]): optimizer string. See `utils.op` for details.
                Defaults to `sgd`.
            optimizer_args (Optional[dict]): optional arguments for optimizer.

        '''
        from .. import _manager as manager
        optimizer = self.optimizer
        optimizer_args = self.optimizer_args or dict()
        
        grad_clip = kwargs.pop('grad_clip', None)

        if optimizer not in _ops.keys():
            raise KeyError('Optimizer `%s` not found, available: %s'
                           % (optimizer, _ops.keys()))

        session = self.session
        optimizer = _ops[optimizer]
    
        if len(model_costs) == 0:
            model_costs = [(None, None)]

        grads = OrderedDict()
        tparams = OrderedDict()

        for models, cost in model_costs:
            
            # Models
            if models is None:
                tparams_ = manager.tparams
            else:
                tparams_ = OrderedDict()
                if isinstance(models, str): models = [models]
                for model in models:
                    for k, v in manager.tparams.items():
                        prefix = '.'.join(k.split('.')[:-1])
                        if model == prefix: tparams_[k] = v
                        
            # Theano parameters
            tparams_ = OrderedDict((k, v)
                for k, v in tparams_.items()
                if (v not in session.updates.keys()) and (k not in self.excludes))
            self.tparams = list(set(self.tparams + tparams_.keys()))
            
            # Costs
            if cost is None:
                self.logger.debug('Using global cost for models %s' % models)
                cost = session.cost
            else:
                self.logger.debug('Using cost `%s` for models %s'
                                  % (cost, models))
                if isinstance(cost, list):
                    cost_ = T.constant(0).astype(floatX)
                    for cost__ in cost:
                        try:
                            cost_ += session.costs[cost__]
                        except KeyError as e:
                            self.logger.error(
                                'Cost `{}` not found, available: {}'.format(
                                    cost__, session.costs.keys()))
                    cost = cost_
                else:
                    try:
                        cost = session.costs[cost]
                    except KeyError as e:
                        self.logger.error(
                            'Cost `{}` not found, available: {}'.format(
                                cost, session.costs.keys()))

            # Gradients
            self.logger.debug('Computing gradients for params: %s' % tparams_.keys())
            try:
                grads_ = T.grad(
                    cost, wrt=tparams_.values(), consider_constant=session.constants)
            except theano.gradient.DisconnectedInputError:
                self.logger.error('DisconnectedInputError')
                for k, v in tparams_.items():
                    self.logger.info('Trying {}'.format(k))
                    T.grad(cost, wrt=v, consider_constant=session.constants)
            grads_ = OrderedDict((k, g) for k, g in zip(tparams_.keys(), grads_))
            for k, v in grads_.items():
                if k in grads.keys():
                    grads[k] += grads_[k]
                else:
                    grads[k] = grads_[k]
                    
            tparams.update(tparams_)
        
        tparams = OrderedDict((k, tparams[k]) for k in grads.keys())
        self.logger.debug('Total params: %s' % tparams.keys())
        
        if grad_clip is not None:
            self.clip_grads(grads, **grad_clip)

        lr = T.scalar(name='lr')
        f_grad, f_update = optimizer(
            lr, tparams, grads, session.inputs, session.cost,
            extra_ups=session.updates, **optimizer_args)

        self.f_grads.append(f_grad)
        self.f_updates.append(f_update)
        self.f_freqs.append(1)
        
    def clip_grads(self, grads, clip_type='minmax', clip_min=-1., clip_max=1.,
                   clip_norm=1., clip_keys=None):
        self.logger.info('Clipping gradients with type {} min/max/norm '
                         '(when applicable): {}/{}/{}'.format(
                            clip_type, clip_min, clip_max, clip_norm))
        for k in grads.keys():
            if clip_keys is not None and k not in clip_keys:
                continue
            if clip_type == 'minmax':
                grads[k] = T.clip(grads[k], clip_min, clip_max)
            elif clip_type == 'norm':
                grads[k] = T.switch(
                    T.gt((grads[k] ** 2).sum(), clip_norm),
                    clip_norm * grads[k] / (grads[k] ** 2).sum(),
                    grads[k])
            else:
                raise TypeError(clip_type)
